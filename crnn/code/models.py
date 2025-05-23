from torch import nn



class CRNN(nn.Module): # Initial model with rnn hidden state aggregation done in a really inefficient way, (blows up the num of paramter)
    def __init__(self, # bug in dropout_specification of gru, do contiguous memory after permutate
                 
                # weight initialization was using their ref 46

                 # for input they had 40 mel bands, time steps varied
                 input_shape, # (Frequecies, Timesteps, Channels)

                 # cnn layers, do grid search 1, 2, 3, 4
                 #      the kernel, best 5, next 3
                 #      the filters used by them 96
                 #      pool sizes 2x1 non overlapping stride
                 #      dropout after cnn was used with 0.25
                 cnn_layers=3,kernels=(5, 5, 5), filters=(96, 96, 96), pool_sizes=(2, 2, 2),
                 cnn_dropout=(0.25, 0.25, 0.25),
                # rnn layers, do grid search 1, 2, 3
                #       their dropout was done in custom with ref 35
                #           I have just used the same constant for prob, and implemented the inbuilt pytorch dropout for rnns
                 rnn_layers=2, rnn_hidden=(256, 256),
                 rnn_dropout=(0.25,),

                # fnn layer hidden units is taken from their ref 21, where it shares a lot of simularity, 
                #           no real basis for the hidden units though, as the arch in ref 21 is very different
                #           do grid search then 128, 256, 512, 1024, 2048, 4096(overkill)
                #       dropout i just used the same constant
                 fnn_layers=1, fnn_hidden=(1024,), fnn_dropout=(0.25,),
                 output_shape=5):
        
        super(CRNN, self).__init__()

        self.time_steps = input_shape[1]

        # =============================================================================

        self.cnn_blocks = nn.Sequential()
        in_channels = input_shape[2]
        freq = input_shape[0]

        for i in range(cnn_layers):
            self.cnn_blocks.add_module(f'cnn_{i}', nn.Conv2d(in_channels, filters[i], kernel_size=kernels[i], padding="same"))
            self.cnn_blocks.add_module(f'relu_{i}', nn.ReLU())
            self.cnn_blocks.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(pool_sizes[i], 1), stride=(pool_sizes[i], 1)))
            self.cnn_blocks.add_module(f'batchnorm_{i}', nn.BatchNorm2d(filters[i]))
            self.cnn_blocks.add_module(f'dropout_{i}', nn.Dropout(cnn_dropout[i]))

            in_channels = filters[i]
            freq = (freq - (pool_sizes[i] - 1) -1)//pool_sizes[i] + 1

        self.freq = freq
        self.channels = filters[-1]

        # ==============================================================================

        self.rnn_input_size = self.freq * self.channels

        self.rnn = nn.Sequential()
        input_size = self.rnn_input_size
        for i in range(rnn_layers):
            self.rnn.add_module(f'gru_{i}', nn.GRU(input_size, rnn_hidden[i], batch_first=True, dropout=rnn_dropout[i] if i < rnn_layers - 1 else 0))
            input_size = rnn_hidden[i]

        # ===============================================================================

        input_size = self.time_steps*rnn_hidden[-1]

        self.fnn_blocks = nn.Sequential()
        for i in range(fnn_layers):
            self.fnn_blocks.add_module(f'fc_{i}', nn.Linear(input_size, fnn_hidden[i]))
            self.fnn_blocks.add_module(f'relu_{i}', nn.ReLU())
            self.fnn_blocks.add_module(f'batchnorm_{i}', nn.BatchNorm1d(fnn_hidden[i]))
            self.fnn_blocks.add_module(f'dropout_{i}', nn.Dropout(fnn_dropout[i]))
            input_size = fnn_hidden[i]

        self.output_layer = nn.Linear(input_size, output_shape)

    def forward(self, x): 

        x = x.permute(0, 3, 1, 2) # (B, C_in, F, T)
        
          
        # print("cnn input: ", x.shape)
        for name, layer in self.cnn_blocks.named_children(): # outs = # (B, C_out, F', T)
            x = layer(x)
            # print(f"{name}: ", x.shape)
        
        # print("after conv: ", x.shape)

        B, C_out, F_prime, T = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, T, C_out, F')

        x = x.reshape(B, T, F_prime * C_out)  # (B, T, F' * C_out)

        # print("for rnn: ", x.shape)

        for name, layer in self.rnn.named_children(): # outs: (B, T, H_out)
            x, _ = layer(x)
            # print(f"{name} ", x.shape)

        B, T, H_out = x.size()
        # print("after rnn: ", x.shape)

        x = x.reshape(B, T*H_out)

        # print("for fnn: ", x.shape)

        x = self.fnn_blocks(x)

        # print("after fnn: ", x.shape)

        x = self.output_layer(x)

        # print("outs: ", x.shape)

        return x
    

#     # taking 40 bands, arbitary time steps
# test = CRNN(input_shape=(40, 128, 1), output_shape=5)
# summary(test, input_size=(64, 40, 128, 1), col_names=['input_size', 'output_size', 'num_params'])

# # NOTE: main contributor to the parameter number is: concatenation of outputs from gru and feeding to linear layer
# # need reduction methodology here




class CRNN2(nn.Module): # Made rnn hidden state aggregation more efficient
    def __init__(self, # bug in dropout_specification of gru, do contiguous memory after permutate
                 
                # weight initialization was using their ref 46

                 # for input they had 40 mel bands, time steps varied
                 input_shape, # (Frequecies, Timesteps, Channels)

                 # cnn layers, do grid search 1, 2, 3, 4
                 #      the kernel, best 5, next 3
                 #      the filters used by them 96
                 #      pool sizes 2x1 non overlapping stride
                 #      dropout after cnn was used with 0.25
                 cnn_layers=3,kernels=(5, 5, 5), filters=(96, 96, 96), pool_sizes=(2, 2, 2),
                 cnn_dropout=(0.25, 0.25, 0.25),
                # rnn layers, do grid search 1, 2, 3
                #       their dropout was done in custom with ref 35
                #           I have just used the same constant for prob, and implemented the inbuilt pytorch dropout for rnns
                 rnn_layers=2, rnn_hidden=(256, 256),
                 rnn_dropout=(0.25,),

                # fnn layer hidden units is taken from their ref 21, where it shares a lot of simularity, 
                #           no real basis for the hidden units though, as the arch in ref 21 is very different
                #           do grid search then 128, 256, 512, 1024, 2048, 4096(overkill)
                #       dropout i just used the same constant
                # encoder out nodes: refers to the output nodes of the encoder layer
                #           the encoder layer is applied on each of the [time_steps] outputs of the rnn layer
                #           effectively transforms from the rnn output from (time_steps, rnn_hidden[-1]) to (time_steps, enc_out_nodes)
                encoder_out_nodes = 16,
                fnn_layers=1, fnn_hidden=(1024,), fnn_dropout=(0.25,),
                output_shape=5):
        
        super(CRNN2, self).__init__()
        self.input_shape = input_shape

        self.time_steps = input_shape[1]

        # =============================================================================

        self.cnn_blocks = nn.Sequential()
        in_channels = input_shape[2]
        freq = input_shape[0]

        for i in range(cnn_layers):
            self.cnn_blocks.add_module(f'cnn_{i}', nn.Conv2d(in_channels, filters[i], kernel_size=kernels[i], padding="same"))
            self.cnn_blocks.add_module(f'relu_{i}', nn.ReLU())
            self.cnn_blocks.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(pool_sizes[i], 1), stride=(pool_sizes[i], 1)))
            self.cnn_blocks.add_module(f'batchnorm_{i}', nn.BatchNorm2d(filters[i]))
            self.cnn_blocks.add_module(f'dropout_{i}', nn.Dropout(cnn_dropout[i]))

            in_channels = filters[i]
            freq = (freq - (pool_sizes[i] - 1) -1)//pool_sizes[i] + 1

        self.freq = freq
        self.channels = filters[-1]

        # ==============================================================================

        self.rnn_input_size = self.freq * self.channels

        self.rnn = nn.Sequential()
        input_size = self.rnn_input_size
        for i in range(rnn_layers):
            self.rnn.add_module(f'gru_{i}', nn.GRU(input_size, rnn_hidden[i], batch_first=True, dropout=rnn_dropout[i] if i < rnn_layers - 1 else 0))
            input_size = rnn_hidden[i]

        # ===============================================================================
        # current shape: (time_steps, h_out)

        input_size = (self.time_steps, rnn_hidden[-1])

        self.encoder_layer = nn.Linear(input_size[1], encoder_out_nodes)

        input_size = self.time_steps*encoder_out_nodes

        self.fnn_blocks = nn.Sequential()
        for i in range(fnn_layers):
            self.fnn_blocks.add_module(f'fc_{i}', nn.Linear(input_size, fnn_hidden[i]))
            self.fnn_blocks.add_module(f'relu_{i}', nn.ReLU())
            self.fnn_blocks.add_module(f'batchnorm_{i}', nn.BatchNorm1d(fnn_hidden[i]))
            self.fnn_blocks.add_module(f'dropout_{i}', nn.Dropout(fnn_dropout[i]))
            input_size = fnn_hidden[i]

        self.output_layer = nn.Linear(input_size, output_shape)

    def forward(self, x): 

        
        if x.shape[1] == self.input_shape[1]: # (B, T, F, C_in) wrong order
            x = x.permute(0, 2, 1, 3) # correct order


        x = x.permute(0, 3, 1, 2) # (B, C_in, F, T) 
        
          
        # print("cnn input: ", x.shape)
        for name, layer in self.cnn_blocks.named_children(): # outs = # (B, C_out, F', T)
            x = layer(x)
            # print(f"{name}: ", x.shape)
        
        # print("after conv: ", x.shape)

        B, C_out, F_prime, T = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, T, C_out, F')

        x = x.reshape(B, T, F_prime * C_out)  # (B, T, F' * C_out)

        # print("for rnn: ", x.shape)

        for name, layer in self.rnn.named_children(): # outs: (B, T, H_out)
            x, _ = layer(x)
            # print(f"{name} ", x.shape)

        B, T, H_out = x.size()
        # print("after rnn: ", x.shape)


        # =====================================

        x = self.encoder_layer(x)

        B, enc_nodes, H_out = x.size()
        # print(x.size())


        x = x.reshape(B, enc_nodes*H_out)

        # print("for fnn: ", x.shape)

        x = self.fnn_blocks(x)

        # print("after fnn: ", x.shape)

        x = self.output_layer(x)

        # print("outs: ", x.shape)

        return x
    
#     # taking 40 bands, arbitary time steps
# test = CRNN2(input_shape=(40, 128, 1), fnn_layers=2, fnn_hidden=(1024,1024), fnn_dropout=(0.25,0.25), output_shape=5)
# summary(test, input_size=(64, 40, 128, 1), col_names=['input_size', 'output_size', 'num_params'])

# # NOTE: main contributor to the parameter number is: concatenation of outputs from gru and feeding to linear layer
# # need reduction methodology here

# # NOTE: That ^ has been reduced as follows: use an encoder layer that goes over each time_step's the gru outputs
# # to reduce the hidden units parameter to encoder_units



class CRNN3(nn.Module): # fixed droput for rnn, added normalization to rnn, also added activation for the encoder layer
    def __init__(self, 
                 
                # weight initialization was using their ref 46

                 # for input they had 40 mel bands, time steps varied
                 input_shape, # (Frequecies, Timesteps, Channels)

                 # cnn layers, do grid search 1, 2, 3, 4
                 #      the kernel, best 5, next 3
                 #      the filters used by them 96
                 #      pool sizes 2x1 non overlapping stride
                 #      dropout after cnn was used with 0.25
                 cnn_layers=3,kernels=(5, 5, 5), filters=(96, 96, 96), pool_sizes=(2, 2, 2),
                 cnn_dropout=(0.25, 0.25, 0.25),
                # rnn layers, do grid search 1, 2, 3
                #       their dropout was done in custom with ref 35
                #           I have just used the same constant for prob, and implemented the inbuilt pytorch dropout for rnns
                 rnn_layers=2, rnn_hidden=(256, 256),
                 rnn_dropout=(0.25, 0.0),

                # fnn layer hidden units is taken from their ref 21, where it shares a lot of simularity, 
                #           no real basis for the hidden units though, as the arch in ref 21 is very different
                #           do grid search then 128, 256, 512, 1024, 2048, 4096(overkill)
                #       dropout i just used the same constant
                # encoder out nodes: refers to the output nodes of the encoder layer
                #           the encoder layer is applied on each of the [time_steps] outputs of the rnn layer
                #           effectively transforms from the rnn output from (time_steps, rnn_hidden[-1]) to (time_steps, enc_out_nodes)
                encoder_out_nodes = 16,
                fnn_layers=1, fnn_hidden=(1024,), fnn_dropout=(0.25,),
                output_shape=5):
        
        super(CRNN3, self).__init__()
        self.input_shape = input_shape

        self.time_steps = input_shape[1]

        # =============================================================================

        self.cnn_blocks = nn.ModuleList()
        in_channels = input_shape[2]
        freq = input_shape[0]

        for i in range(cnn_layers):
            self.cnn_blocks.add_module(f'cnn_{i}', nn.Conv2d(in_channels, filters[i], kernel_size=kernels[i], padding="same"))
            self.cnn_blocks.add_module(f'relu_{i}', nn.ReLU())
            self.cnn_blocks.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(pool_sizes[i], 1), stride=(pool_sizes[i], 1)))
            self.cnn_blocks.add_module(f'batchnorm_{i}', nn.BatchNorm2d(filters[i]))
            self.cnn_blocks.add_module(f'dropout_{i}', nn.Dropout(cnn_dropout[i]))
            

            in_channels = filters[i]
            freq = (freq - (pool_sizes[i] - 1) -1)//pool_sizes[i] + 1

        self.freq = freq
        self.channels = filters[-1]

        # ==============================================================================

        self.rnn_input_size = self.freq * self.channels

        self.rnn = nn.ModuleList()
        input_size = self.rnn_input_size
        for i in range(rnn_layers):
            self.rnn.add_module(f'gru_{i}', nn.GRU(input_size, rnn_hidden[i], batch_first=True))
            self.rnn.add_module(f'norm_{i}', nn.LayerNorm(rnn_hidden[i]))
            self.rnn.add_module(f'drop_{i}', nn.Dropout(rnn_dropout[i]))
            input_size = rnn_hidden[i]

        # ===============================================================================
        # current shape: (time_steps, h_out)

        input_size = (self.time_steps, rnn_hidden[-1])

        self.encoder_layer = nn.Sequential(nn.Linear(input_size[1], encoder_out_nodes), nn.ReLU())

        input_size = self.time_steps*encoder_out_nodes

        self.fnn_blocks = nn.Sequential()
        for i in range(fnn_layers):
            self.fnn_blocks.add_module(f'fc_{i}', nn.Linear(input_size, fnn_hidden[i]))
            self.fnn_blocks.add_module(f'relu_{i}', nn.ReLU())
            self.fnn_blocks.add_module(f'batchnorm_{i}', nn.BatchNorm1d(fnn_hidden[i]))
            self.fnn_blocks.add_module(f'dropout_{i}', nn.Dropout(fnn_dropout[i]))

            input_size = fnn_hidden[i]

        self.output_layer = nn.Linear(input_size, output_shape)

    def forward(self, x): 

        
        if x.shape[1] == self.input_shape[1]: # (B, T, F, C_in) wrong order
            x = x.permute(0, 2, 1, 3) # correct order


        x = x.permute(0, 3, 1, 2) # (B, C_in, F, T) 
        
        
        # print("cnn input: ", x.shape)
        for name, layer in self.cnn_blocks.named_children(): # outs = # (B, C_out, F', T)
            x = layer(x)
            # print(f"{name}: ", x.shape)
        
        # print("after conv: ", x.shape)

        B, C_out, F_prime, T = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, T, C_out, F')

        x = x.reshape(B, T, F_prime * C_out)  # (B, T, F' * C_out)

        # print("for rnn: ", x.shape)

        for name, layer in self.rnn.named_children(): # outs: (B, T, H_out)
            if isinstance(layer, nn.GRU):
                x, _ = layer(x) # discard the layer layers output
            elif isinstance(layer, nn.LayerNorm):
                x = layer(x)
            elif isinstance(layer, nn.Dropout):
                x = layer(x)
            
            # print(f"{name} ", x.shape)

        B, T, H_out = x.size()
        # print("after rnn: ", x.shape)


        # =====================================

        x = self.encoder_layer(x)

        B, T, H_out_enc = x.size()
        # print(x.size())


        x = x.reshape(B, T*H_out_enc)

        # print("for fnn: ", x.shape)

        x = self.fnn_blocks(x)

        # print("after fnn: ", x.shape)

        x = self.output_layer(x)

        # print("outs: ", x.shape)

        return x





class CRNN4(nn.Module): # the last rnn layer's last hidden units are directly sent to the fnn, it doesn't take the all the hidden units of all timesteps of the last rnn layer now. In view of this the encoder (which reduced the (timesteps, hidden) to (timesteps, small_hidden) is removed)
    def __init__(self, 
                 
                # weight initialization was using their ref 46

                 # for input they had 40 mel bands, time steps varied
                 input_shape, # (Frequecies, Timesteps, Channels)

                 # cnn layers, do grid search 1, 2, 3, 4
                 #      the kernel, best 5, next 3
                 #      the filters used by them 96
                 #      pool sizes 2x1 non overlapping stride
                 #      dropout after cnn was used with 0.25
                 cnn_layers=3,kernels=(5, 5, 5), filters=(96, 96, 96), pool_sizes=(2, 2, 2),
                 cnn_dropout=(0.25, 0.25, 0.25),
                # rnn layers, do grid search 1, 2, 3
                #       their dropout was done in custom with ref 35
                #           I have just used the same constant for prob, and implemented the inbuilt pytorch dropout for rnns
                 rnn_layers=2, rnn_hidden=(256, 256),
                 rnn_dropout=(0.25, 0.0),

                # fnn layer hidden units is taken from their ref 21, where it shares a lot of simularity, 
                #           no real basis for the hidden units though, as the arch in ref 21 is very different
                #           do grid search then 128, 256, 512, 1024, 2048, 4096(overkill)
                #       dropout i just used the same constant
                fnn_layers=1, fnn_hidden=(1024,), fnn_dropout=(0.25,),
                output_shape=5):
        
        super(CRNN4, self).__init__()
        self.input_shape = input_shape

        self.time_steps = input_shape[1]

        # =============================================================================

        self.cnn_blocks = nn.ModuleList()
        in_channels = input_shape[2]
        freq = input_shape[0]

        for i in range(cnn_layers):
            self.cnn_blocks.add_module(f'cnn_{i}', nn.Conv2d(in_channels, filters[i], kernel_size=kernels[i], padding="same"))
            self.cnn_blocks.add_module(f'relu_{i}', nn.ReLU())
            self.cnn_blocks.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(pool_sizes[i], 1), stride=(pool_sizes[i], 1)))
            self.cnn_blocks.add_module(f'batchnorm_{i}', nn.BatchNorm2d(filters[i]))
            self.cnn_blocks.add_module(f'dropout_{i}', nn.Dropout(cnn_dropout[i]))

            in_channels = filters[i]
            freq = (freq - (pool_sizes[i] - 1) -1)//pool_sizes[i] + 1

        self.freq = freq
        self.channels = filters[-1]

        # ==============================================================================

        self.rnn_input_size = self.freq * self.channels

        self.rnn = nn.ModuleList()
        input_size = self.rnn_input_size
        for i in range(rnn_layers):
            self.rnn.add_module(f'gru_{i}', nn.GRU(input_size, rnn_hidden[i], batch_first=True))
            self.rnn.add_module(f'norm_{i}', nn.LayerNorm(rnn_hidden[i]))
            self.rnn.add_module(f'drop_{i}', nn.Dropout(rnn_dropout[i]))
            input_size = rnn_hidden[i]

        # ===============================================================================
        # current shape: (time_steps, h_out)

        input_size = rnn_hidden[-1]

        # current shape: (h_out)
        self.fnn_blocks = nn.Sequential()
        for i in range(fnn_layers):
            self.fnn_blocks.add_module(f'fc_{i}', nn.Linear(input_size, fnn_hidden[i]))
            self.fnn_blocks.add_module(f'relu_{i}', nn.ReLU())
            self.fnn_blocks.add_module(f'batchnorm_{i}', nn.BatchNorm1d(fnn_hidden[i]))
            self.fnn_blocks.add_module(f'dropout_{i}', nn.Dropout(fnn_dropout[i]))
            input_size = fnn_hidden[i]

        self.output_layer = nn.Linear(input_size, output_shape)

    def forward(self, x): 

        
        if x.shape[1] == self.input_shape[1]: # (B, T, F, C_in) wrong order
            x = x.permute(0, 2, 1, 3) # correct order


        x = x.permute(0, 3, 1, 2) # (B, C_in, F, T) 
        
        
        # print("cnn input: ", x.shape)
        for name, layer in self.cnn_blocks.named_children(): # outs = # (B, C_out, F', T)
            x = layer(x)
            # print(f"{name}: ", x.shape)
        
        # print("after conv: ", x.shape)

        B, C_out, F_prime, T = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, T, C_out, F')

        x = x.reshape(B, T, F_prime * C_out)  # (B, T, F' * C_out)

        # print("for rnn: ", x.shape)

        for name, layer in self.rnn.named_children(): # outs: (B, T, H_out)
            if isinstance(layer, nn.GRU):
                x, _ = layer(x) # discard the layer layers output
            elif isinstance(layer, nn.LayerNorm):
                x = layer(x)
            elif isinstance(layer, nn.Dropout):
                x = layer(x)
            
            # print(f"{name} ", x.shape)
        


        # =====================================

        x = x[:, -1, :]
        B, D = x.size()


        # print("for fnn: ", x.shape)

        x = self.fnn_blocks(x)

        # print("after fnn: ", x.shape)

        x = self.output_layer(x)

        # print("outs: ", x.shape)

        return x