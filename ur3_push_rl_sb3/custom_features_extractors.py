import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GRUExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=32):
        assert isinstance(observation_space, gym.spaces.Dict)
        input_shape = 0
        for key in observation_space.keys():
            if key in ["observation", "achieved_goal"]:
                input_shape += observation_space[key].shape[1]
            else:
                input_shape += observation_space[key].shape[0]
        
        super().__init__(observation_space, features_dim=features_dim)

        self.gru = nn.GRU(input_size=input_shape, 
                          hidden_size=features_dim,
                          num_layers=1,
                          batch_first=True)
                
    def forward(self, observations):
        # shape: batch_size x sequence_length x input_dim
        batch_size = observations["observation"].shape[0]
        seq_lengths = torch.sum(torch.sum(observations["observation"], dim=-1) != 0, dim=1)
        seq_lengths_u = seq_lengths.unique()

        h_all = torch.zeros(batch_size, self.gru.hidden_size).to(seq_lengths.device)
        for sl in seq_lengths_u:
            mask_sl = seq_lengths == sl
            # concatenate dict values
            tensor_list = []
            for key in observations.keys():
                if key in ["observation", "achieved_goal"]:
                    tensor_list.append(observations[key][mask_sl, :sl, :])
                else:
                    tensor_list.append(torch.unsqueeze(observations[key][mask_sl,:], dim=1).repeat(1,sl,1))
            vec_obs = torch.cat(tensor_list, dim=2)
            # get hidden states
            _, h = self.gru(vec_obs)
            h_all[mask_sl,:] = h[0,:,:]

        return h_all
    
if __name__ == "__main__":
    rnn = nn.GRU(10, 20, 1)
    input = torch.randn(5, 2, 10)
    h0 = torch.randn(1, 2, 20)
    output, hn = rnn(input, h0)

    print(torch.equal(output[-1:,:,:], hn))
    print(output.shape, hn.shape)
    