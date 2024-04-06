import torch
import torch.nn as nn

class CharRNN(nn.Module):
    # ... existing code ...

    def forward(self, input_seq, hidden=None):
        # Pass the whole input sequence through the nn.Embedding layer
        embed = self.embedding_layer(input_seq)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).to(input_seq.device)
        
        # Initialize outputs tensor
        outputs = torch.zeros(len(input_seq), self.n_chars).to(input_seq.device)
        
        # Iterate over the input sequence
        for i in range(len(input_seq)):
            # Apply the rnn_cell
            output, hidden = self.rnn_cell(embed[i], hidden)
            
            # Collect the output of the rnn_cell at each timestep
            outputs[i] = output
        
        return outputs, hidden

    def top_k_filtering(self, logits, k):
        # Sort the logits
        values, _ = torch.topk(logits, k)
        
        # Create a mask for values that don't meet the criteria
        mask = logits < values[:, [-1]]
        
        # Set the values in the tensor that don’t meet the criteria to negative infinity
        logits[mask] = float('-inf')
        
        return logits

    def top_p_filtering(self, logits, p):
        # Sort the logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Create a mask for values that don't meet the criteria
        mask = cumulative_probs > p
        
        # Keep the first token above the threshold
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = 0
        
        # Set the values in the tensor that don’t meet the criteria to negative infinity
        sorted_logits[mask] = float('-inf')
        
        # Restore the logits to their original order
        _, inverse_indices = torch.sort(sorted_indices, dim=-1)
        logits = sorted_logits.gather(dim=-1, index=inverse_indices)
        
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class CharRNN(nn.Module):
    # ... existing code ...

    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        # Initialize the sequence with the starting character
        sequence = [starting_char]
        
        # Initialize hidden state
        hidden = None
        
        # Iterate over the sequence length
        for _ in range(seq_len):
            # Get the last character in the sequence
            input_seq = torch.tensor([sequence[-1]]).to('cuda')
            
            # Pass the input through the network
            output, hidden = self.forward(input_seq, hidden)
            
            # Apply temperature
            output = output / temp
            
            # Apply top-K and top-P filtering if specified
            if top_k is not None:
                output = self.top_k_filtering(output, top_k)
            if top_p is not None:
                output = self.top_p_filtering(output, top_p)
            
            # Softmax the output to get probabilities
            probabilities = F.softmax(output, dim=-1)
            
            # Sample from the distribution
            distribution = Categorical(probabilities)
            next_char = distribution.sample()
            
            # Add the sampled character to the sequence
            sequence.append(next_char.item())
        
        return sequence

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)
    
    
import torch

def train(model, dataset, lr, out_seq_len, num_epochs):
    # Initialize optimizer and loss function
    optimizer = model.get_optimizer(lr)
    loss_fn = model.get_loss_function()

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n = 0
    running_loss = 0
    for epoch in range(num_epochs):
        for in_seq, out_seq in dataset.get_example():
            # Move tensors to the device
            in_seq = in_seq.to(device)
            out_seq = out_seq.to(device)

            # Initialize hidden state to None for each training iteration
            hidden = None

            # Main loop code
            output, _ = model(in_seq, hidden)
            loss = loss_fn(output.view(-1, model.n_chars), out_seq.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n += 1

        # Print info every X examples
        print(f"Epoch {epoch}. Running loss so far: {(running_loss/n):.8f}")

        print("\n-------------SAMPLE FROM MODEL-------------")

        # Code to sample a sequence from your model randomly
        with torch.no_grad():
            starting_char = torch.randint(model.n_chars, (1,))
            generated_seq = model.sample_sequence(starting_char, out_seq_len)
            print(''.join([dataset.int2char[i] for i in generated_seq]))

        print("\n------------/SAMPLE FROM MODEL/------------")

        n = 0
        running_loss = 0

    return model


# Create your model
model = CharRNN(n_chars, embedding_size, hidden_size)

# Load your datasets
sherlock_dataset = TextDataset('sherlock.txt')
shakespeare_dataset = TextDataset('shakespeare.txt')

# Train your model on the Sherlock dataset
train(model, sherlock_dataset, lr, out_seq_len, num_epochs)

# Generate samples with different temperature values
for temp in [0.2, 0.5, 1.0, 1.2]:
    print(f"\nTemperature: {temp}")
    starting_char = torch.randint(model.n_chars, (1,))
    print(''.join([sherlock_dataset.int2char[i] for i in model.sample_sequence(starting_char, 100, temp=temp)]))

# Train your model on the Shakespeare dataset
train(model, shakespeare_dataset, lr, out_seq_len, num_epochs)

# Generate samples with different temperature values
for temp in [0.2, 0.5, 1.0, 1.2]:
    print(f"\nTemperature: {temp}")
    starting_char = torch.randint(model.n_chars, (1,))
    print(''.join([shakespeare_dataset.int2char[i] for i in model.sample_sequence(starting_char, 100, temp=temp)]))

