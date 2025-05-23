import os
import json
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
import csv
import time
from torch import nn
import pickle
import pandas as pd

def load_config(config_path):
    """Load the configuration file containing preprocessing parameters."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_fbank_directory(wav_dir):
    """Create a directory for storing mel spectrograms."""
    wav_path = Path(wav_dir)
    fbank_dir = wav_path.parent / 'fbank'
    fbank_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories if they don't exist
    return fbank_dir

def preprocess_audio(wav_path, config):
    """Preprocess a single audio file to generate mel spectrogram.
    
    Args:
        wav_path (str): Path to the wav file
        config (dict): Configuration dictionary containing preprocessing parameters
        
    Returns:
        torch.Tensor: Preprocessed mel spectrogram
    """
    # Load audio
    waveform, sr = torchaudio.load(wav_path)
    
    # Normalize waveform
    waveform = waveform - waveform.mean()
    
    # Generate mel spectrogram
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=config['num_mel_bins'],
        dither=0.0,
        frame_shift=10
    )
    
    # Pad or cut to target length
    target_length = config['target_length']
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    print(f"og size of fbank: {fbank.shape} ", end='   ')
    if p > 0:
        # Pad with zeros
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
        print(f"padding, new shape: {fbank.shape}")
    elif p < 0:
        # Cut to target length
        fbank = fbank[0:target_length, :]
        print(f"curring, new shape: {fbank.shape}")
    
    # Normalize using dataset statistics
    if not config.get('skip_norm', False):
        fbank = (fbank - config['mean']) / (config['std'] * 2)
    
    return fbank

def generate_fbank_json(wav_json_path, fbank_dir, output_json_path):
    """Generate a new JSON file pointing to fbank files instead of wav files.
    
    Args:
        wav_json_path (str): Path to the original JSON file containing wav file paths
        fbank_dir (str): Directory containing the fbank files
        output_json_path (str): Path where the new JSON file will be saved
    """
    # Load the original JSON file
    with open(wav_json_path, 'r') as f:
        data = json.load(f)
    
    # Create new data structure
    new_data = {
        'data': [],
        'metadata': data.get('metadata', {})  # Preserve metadata if it exists
    }
    
    # Process each entry in the original JSON
    for item in tqdm(data['data'], desc="Generating fbank JSON"):
        # Get the wav path and convert it to fbank path
        wav_path = Path(item['wav'])
        rel_path = wav_path.relative_to(Path(wav_path).parent)
        fbank_path = Path(fbank_dir) / rel_path.with_suffix('.pt')
        
        # Create new item with fbank path
        new_item = item.copy() # leave label info as it is
        new_item['wav'] = str(fbank_path)
        new_data['data'].append(new_item)
    
    # Save the new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"Generated fbank JSON file at {output_json_path}")

def process_dataset(wav_dir, config_path, config_dict=None):
    """Process all wav files in the directory to generate mel spectrograms.
    
    Args:
        wav_dir (str): Directory containing wav files
        config_path (str): Path to the configuration file
        config_dict (dict, optional): Dictionary containing audio configuration parameters.
            If provided and config_path doesn't exist, this will be used instead.
    """
    # Load configuration
    if os.path.exists(config_path):
        config = load_config(config_path)
    elif config_dict is not None:
        config = config_dict
    else:
        raise ValueError("Either config_path must exist or config_dict must be provided")
    
    # Create fbank directory
    fbank_dir = create_fbank_directory(wav_dir)
    
    # Get all wav files
    wav_files = list(Path(wav_dir).glob('**/*.wav'))
    
    print(f"Processing {len(wav_files)} wav files...")
    
    # Process each wav file
    for wav_path in tqdm(wav_files):
        try:
            # Generate relative path for fbank file
            rel_path = wav_path.relative_to(wav_dir)
            fbank_path = fbank_dir / rel_path.with_suffix('.pt')
            
            # Create parent directories if they don't exist
            fbank_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate and save mel spectrogram
            fbank = preprocess_audio(str(wav_path), config)
            torch.save(fbank, fbank_path)
            
        except Exception as e:
            print(f"Error processing {wav_path}: {str(e)}")
            continue
    
    print(f"Preprocessing complete. Mel spectrograms saved to {fbank_dir}")
    
    # Generate fbank JSON files for train, eval, and test sets if they exist
    data_dir = Path(wav_dir).parent
    json_files = {
        'train': data_dir / 'datafiles' / 'esc_train_data.json',
        'eval': data_dir / 'datafiles' / 'esc_eval_data.json',
        'test': data_dir / 'datafiles' / 'esc_test_data.json'
    }
    
    for split, json_path in json_files.items():
        if json_path.exists():
            output_json = data_dir / 'datafiles' / f'esc_{split}_fbank_data.json'
            generate_fbank_json(str(json_path), str(fbank_dir), str(output_json))

class FbankDataset(Dataset):
    """Dataset class for preprocessed fbank files.
    
    This dataset class works with the preprocessed fbank files generated by process_dataset.
    It loads the preprocessed mel spectrograms directly from disk instead of computing them on the fly.
    
    Args:
        dataset_json_file (str): Path to the JSON file containing fbank file paths and labels
        label_csv (str, optional): Path to the CSV file containing class labels
        audio_conf (dict, optional): Dictionary containing audio configuration parameters
    """
    def __init__(self, dataset_json_file, label_csv=None, audio_conf=None):
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        
        self.data = data_json['data']
        self.audio_conf = audio_conf or {}
        
        # Load label mapping if provided
        self.index_dict = {}
        if label_csv:
            with open(label_csv, 'r') as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    self.index_dict[row['mid']] = row['index']
            self.label_num = len(self.index_dict)
        else:
            self.label_num = None
    
    def __getitem__(self, index):
        """Get a single item from the dataset.
        
        Args:
            index (int): Index of the item to get
            
        Returns:
            tuple: (fbank, label_indices)
                - fbank (torch.Tensor): The preprocessed mel spectrogram
                - label_indices (torch.Tensor): The label indices
        """
        datum = self.data[index]
        
        # Load the preprocessed fbank
        fbank = torch.load(datum['wav'])
        
        # Initialize label indices
        if self.label_num is not None:
            label_indices = np.zeros(self.label_num)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0
            label_indices = torch.FloatTensor(label_indices)
        else:
            label_indices = torch.tensor(0)  # Dummy tensor if no labels
        
        return fbank, label_indices
    
    def __len__(self):
        """Get the total number of items in the dataset."""
        return len(self.data)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    """Train for one epoch."""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    
    end = time.time()
    print("Initiating training batches")
    for i, (audio_input, labels) in enumerate(train_loader):
        # print(f"{i}/{len(train_loader)} | ", end="")
        data_time.update(time.time() - end)
        
        # Move data to device
        audio_input = audio_input.to(device)
        labels = labels.to(device)
        
        # Compute output
        audio_output = model(audio_input)
        
        # Compute loss
        if isinstance(criterion, nn.CrossEntropyLoss):
            loss = criterion(audio_output, torch.argmax(labels.long(), axis=1))
        else:
            loss = criterion(audio_output, labels)
        
        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        loss_meter.update(loss.item(), audio_input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print statistics
        if i % args['print_freq'] == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})')
    
    return loss_meter.avg

def validate(model, val_loader, criterion, device, args):
    """Validate the model."""
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        end = time.time()
        for i, (audio_input, labels) in enumerate(val_loader):
            # Move data to device
            audio_input = audio_input.to(device)
            labels = labels.to(device)
            
            # Compute output
            audio_output = model(audio_input)
            
            # Compute loss
            if isinstance(criterion, nn.CrossEntropyLoss):
                loss = criterion(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = criterion(audio_output, labels)
            
            # Store predictions and targets
            predictions.append(audio_output.cpu())
            targets.append(labels.cpu())
            
            # Update statistics
            loss_meter.update(loss.item(), audio_input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args['print_freq'] == 0:
                print(f'Validation: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})')
    
    # Concatenate predictions and targets
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    
    # Calculate metrics
    if isinstance(criterion, nn.CrossEntropyLoss):
        accuracy = (torch.argmax(predictions, dim=1) == torch.argmax(targets, dim=1)).float().mean()
        metrics = {'accuracy': accuracy.item(), 'loss': loss_meter.avg}
    else:
        # For binary/multi-label classification
        predictions = torch.sigmoid(predictions)
        metrics = {
            'loss': loss_meter.avg,
            'accuracy': ((predictions > 0.5) == targets).float().mean().item()
        }
    
    return metrics

def train_model(model, train_loader, val_loader, args):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Move model to device
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Define loss function
    if args['loss'] == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif args['loss'] == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Unknown loss function: {args["loss"]}')
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    # Define learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=args.lr_patience, verbose=True
    # )

    # ideally for ESC
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args['lrscheduler_start'], 1000, args['lrscheduler_step'])),gamma=args['lrscheduler_decay'])


    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args['epochs']):
        print(f'\nEpoch {epoch+1}/{args["epochs"]}')
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, args)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        best_val_loss = min(val_metrics['loss'], best_val_loss)
        
        if is_best:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }, os.path.join(args['exp_dir'], 'best_model.pth'))
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy']
        }, os.path.join(args['exp_dir'], 'latest_model.pth'))
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_metrics["loss"]:.4f}')
        print(f'Val Accuracy: {val_metrics["accuracy"]:.4f}')

def run_inference(model, val_json_path, label_csv_path, output_csv_path, device=None):
    """Run inference on a validation set and save predictions to CSV.
    
    Args:
        model (nn.Module): The trained model
        val_json_path (str): Path to the validation JSON file containing fbank file paths
        label_csv_path (str): Path to the CSV file containing class labels
        output_csv_path (str): Path where the output CSV file will be saved
        device (torch.device, optional): Device to run inference on. If None, will use CUDA if available.
    
    Returns:
        float: Overall accuracy of the model on the validation set
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load validation data
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
    
    # Load label mapping
    label_map = {}
    with open(label_csv_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            label_map[row['index']] = row['mid']
    num_classes = len(label_map)
    
    # Prepare output CSV
    output_columns = ['file_path'] + [f'prob_class_{i}' for i in range(num_classes)] + ['true_label', 'predicted_label', 'correct']
    output_data = []
    
    # Set model to evaluation mode
    model = model.to(device)
    model.eval()
    
    correct_predictions = 0
    total_predictions = 0
    
    # Process each file
    for item in tqdm(val_data['data'], desc="Running inference"):
        # Load fbank
        fbank = torch.load(item['wav'])
        fbank = fbank.unsqueeze(0).to(device)  # Add batch dimension
        
        # Get true label
        true_label = item['labels'].split(',')[0]  # Assuming single label per file
        
        # Run inference
        with torch.no_grad():
            logits = model(fbank)
            probabilities = torch.softmax(logits, dim=1)
        
        # Get predicted class
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_label = label_map[str(predicted_class)]
        
        # Check if prediction is correct
        is_correct = predicted_label == true_label
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        # Prepare row for CSV
        row = [item['wav']]  # File path
        row.extend(probabilities[0].cpu().numpy())  # Probabilities for each class
        row.extend([true_label, predicted_label, is_correct])
        output_data.append(row)
    
    # Calculate overall accuracy
    accuracy = correct_predictions / total_predictions
    
    # Save results to CSV
    df = pd.DataFrame(output_data, columns=output_columns)
    df.to_csv(output_csv_path, index=False)
    
    print(f"\nInference complete!")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Results saved to: {output_csv_path}")
    
    return accuracy

