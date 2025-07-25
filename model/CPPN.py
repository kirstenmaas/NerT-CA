import torch
import torch.nn as nn

from .Encoder import FourierEncoding, NoEncoding, FreeEncoding

class CPPN(nn.Module):
    def __init__(self, model_definition: dict) -> None:
        super().__init__()
        self.version = "v0.00"
        self.model_definition = model_definition
        self.device = model_definition['device']

        # Architecture parameters
        self.num_early_layers = model_definition['num_early_layers']
        self.num_late_layers = model_definition['num_late_layers']
        self.num_filters = model_definition['num_filters']
        self.num_input_channels = model_definition['num_input_channels'] # x,y,z
        self.num_output_channels = model_definition['num_output_channels']
        self.use_bias = model_definition['use_bias']
        self.use_pos_enc = model_definition['pos_enc']
        self.act_func = model_definition['act_func']

        num_filters = self.num_filters
        use_bias = self.use_bias
        num_output_channels = self.num_output_channels

        # Activation functions
        self.first_act_func = nn.ReLU()
        self.act_func = nn.ReLU()

        # Setup position encoding module
        self.position_encoder = NoEncoding(self.num_input_channels)
        if self.use_pos_enc != 'none':
            self.pos_enc_basis = model_definition['pos_enc_basis']
            if self.use_pos_enc == 'fourier':
                self.position_encoder = FourierEncoding(self.num_input_channels, self.pos_enc_basis, model_definition['fourier_sigma'], self.device)
            elif self.use_pos_enc == 'free_windowed':
                self.position_encoder = FreeEncoding(self.num_input_channels, self.pos_enc_basis, model_definition['pos_enc_window_start'], self.device)

        # Build early layers (MLP before skip connection)
        early_pts_layers = []
        early_pts_layers += self.__create_layer(self.position_encoder.encoding_size, num_filters,
                                           use_bias, activation=self.first_act_func)
        for _ in range(self.num_early_layers):
            early_pts_layers += self.__create_layer(num_filters, num_filters,
                                               use_bias, activation=self.act_func)

        self.early_pts_layers = nn.ModuleList(early_pts_layers)

        # Build late layers (after skip connection, if applicable)
        if self.num_late_layers > 0:
            # Skip connection merges encoded input and early output
            self.skip_connection = self.__create_layer(num_filters + self.position_encoder.encoding_size, num_filters,
                                                use_bias, activation=self.act_func)

            # Late layers following skip connection
            late_pts_layers = []
            for _ in range(self.num_late_layers - 1):
                late_pts_layers += self.__create_layer(num_filters, num_filters,
                                                use_bias, activation=self.act_func)

            self.late_pts_layers = nn.ModuleList(late_pts_layers)

        # Output layer maps to the number of output channels
        self.output_linear = self.__create_layer(num_filters, num_output_channels,
                                        use_bias, activation=None)
        
        # Optionally store activations
        self.store_activations = False
        self.activation_dictionary = {}

    @staticmethod
    def __create_layer(num_in_filters: int, num_out_filters: int,
                       use_bias: bool, activation=nn.ReLU(), dropout=0.5) -> nn.Sequential:
        # Creates a single linear + activation layer
        block = []
        block.append(nn.Linear(num_in_filters, num_out_filters, bias=use_bias)) # Dense layer
        if activation:
            block.append(activation)
        block = nn.Sequential(*block)

        return block

    def activations(self, store_activations: bool) -> None:
        # Toggle activation storage (used for visualization or debugging)
        self.store_activations = store_activations

        if not store_activations:
            self.activation_dictionary = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through encoding and MLP
        input_pts = x
        values = input_pts
        pts_encoded = self.position_encoder(input_pts)

        values = pts_encoded
        for _, pts_layer in enumerate(self.early_pts_layers):
            values = pts_layer(values)

        if self.num_late_layers > 0:
            # Apply skip connection with input encoding
            values = self.skip_connection(torch.cat([pts_encoded, values], dim=-1))

            for _, pts_layer in enumerate(self.late_pts_layers):
                values = pts_layer(values)
        
        # Final output layer
        outputs = self.output_linear(values)
        return outputs

    def update_freq_mask_alpha(self, current_iter, max_iter):
        # Update windowing parameter for FreeEncoding (progressive encoding)
        self.position_encoder.update_alpha(current_iter, max_iter)

    def save(self, filename: str, training_information: dict) -> None:
        # Save the model, parameters, and optional encoding state
        save_parameters = {
                'version': self.version,
                'parameters': self.model_definition,
                'training_information': training_information,
                'model': self.state_dict(), 
            }
        
        if 'free_windowed' in self.use_pos_enc:
            save_parameters['freq_mask_alpha'] = self.position_encoder.alpha

        torch.save(
            save_parameters,
            f=filename)