from tools import (
    get_config
)
from models import get_model
import numpy as np
import os
from scipy import interpolate
import matplotlib.pyplot as plt
import pdb

def visualize_single_sequence(
    model, 
    sequence,
    save_name
):
    model_output = model.predict(sequence)
    print(
        'model prediction for this sequence: {}'.format(model_output.logits_or_values.squeeze(-1).item())
    )
    heat_map = model_output.heat_map.cpu().squeeze(0).numpy()
    
    for c in range(heat_map.shape[-1]):
        # draw attention map for each class
        class_heat_map = heat_map[..., c]
        x = np.arange(len(class_heat_map))
        x_smooth = np.linspace(x.min(), x.max(), len(sequence))
        heat_map_smooth = interpolate.make_interp_spline(x, class_heat_map)(x_smooth)
        heat_map_smooth = np.where(heat_map_smooth < 0, 0, heat_map_smooth)
        heat_map_smooth /= heat_map_smooth.sum()
        heat_map_smooth_cum = heat_map_smooth.cumsum()
        heat_map_smooth /= (heat_map_smooth.max() * 5)

        plt.figure(figsize=(20, 5), dpi=100)
        plt.plot(heat_map_smooth, color='dodgerblue', linewidth=1)
        plt.fill_between(
            np.arange(len(x_smooth)), 
            heat_map_smooth, 
            0, 
            facecolor='dodgerblue', 
            alpha=0.7
        )
        plt.plot(heat_map_smooth_cum, color='darkorange', linewidth=1)
        plt.ylabel('Attention Weight')
        plt.xlabel('Sequence')
        plt.ylim(bottom=0, top=1)
        plt.xlim(left=0, right=len(heat_map_smooth))
        plt.savefig(os.path.join(save_name+f'_class_{c}.png'))
        plt.close()

def main():
    config = get_config()
    model = get_model(config)
    sequence = 'GTCCCTCCCTCCCTCCCTCTTTCTCTCTCCCCGCCTTCCCGGTGCTGCCACAGTGGCGGGTCGGAGGATCCCGGCCAGTCAGCTCGTTCCTGCCTCGCGCCCCTCCTCCCCCCAAAATAAACAGCCCTCCCCACGCCTCGGGAGCCCGGACCGCCCCCTTCCTCCCTCGGCCGGCGTGCTGTGAGGGAGAGTGTGAGCCAGCGCGGCCGGTGGGCGAGCTCCGGGAGTGCGAGAGGCGGGGCGGCGCGGCGGGCCGGGCCGGGTTTCGCGTGTGCGTGCGGGGGCGCGAGCGAGTGCTCATCACGACCCGCCAGTGGCCCCGCGCCGACAACCCGGCGGGCGGGCGGGCGGGGAGCAAGCGGTCAGGCAAGGAGCGGCGGCCGCGAGGCGCGGAGGGGCGCGGAGTCCGCTCGCGCGCACGCACGCACGCACGCGGGGGCGGGGGCAGGCGCGCGCCCGCCTGTCGCGACAGTCGGGGCCGAGGCCCAGGGGGAGGTGGC'
    visualize_single_sequence(model, sequence, 'example')
    
if __name__ == '__main__':
    main()