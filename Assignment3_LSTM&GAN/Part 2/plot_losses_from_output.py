"""
Extract loss values from training output and create plots.
This script parses the training output to extract D loss and G loss values.
"""

import re
import matplotlib.pyplot as plt
import numpy as np

# Training output from the notebook (paste the output here)
training_output = """
[Epoch 0/100] [Batch 0/938] [D loss: 1.3870] [G loss: 0.6953]
[Epoch 0/100] [Batch 500/938] [D loss: 1.3587] [G loss: 0.9767]
[Epoch 1/100] [Batch 62/938] [D loss: 1.3153] [G loss: 0.6809]
[Epoch 1/100] [Batch 562/938] [D loss: 1.2088] [G loss: 0.9609]
[Epoch 2/100] [Batch 124/938] [D loss: 1.2083] [G loss: 0.9580]
[Epoch 2/100] [Batch 624/938] [D loss: 1.1816] [G loss: 1.3066]
[Epoch 3/100] [Batch 186/938] [D loss: 1.4240] [G loss: 0.5783]
[Epoch 3/100] [Batch 686/938] [D loss: 1.3102] [G loss: 0.9163]
[Epoch 4/100] [Batch 248/938] [D loss: 1.1712] [G loss: 1.2451]
[Epoch 4/100] [Batch 748/938] [D loss: 1.1743] [G loss: 1.0371]
[Epoch 5/100] [Batch 310/938] [D loss: 1.2922] [G loss: 0.7368]
[Epoch 5/100] [Batch 810/938] [D loss: 1.1942] [G loss: 0.9553]
[Epoch 6/100] [Batch 372/938] [D loss: 1.2432] [G loss: 0.7191]
[Epoch 6/100] [Batch 872/938] [D loss: 1.2134] [G loss: 0.9392]
[Epoch 7/100] [Batch 434/938] [D loss: 1.4025] [G loss: 1.8186]
[Epoch 7/100] [Batch 934/938] [D loss: 1.2212] [G loss: 0.9622]
[Epoch 8/100] [Batch 496/938] [D loss: 1.2921] [G loss: 0.7118]
[Epoch 9/100] [Batch 58/938] [D loss: 1.2994] [G loss: 1.0610]
[Epoch 9/100] [Batch 558/938] [D loss: 1.2814] [G loss: 0.6378]
[Epoch 10/100] [Batch 120/938] [D loss: 1.3219] [G loss: 0.9974]
[Epoch 10/100] [Batch 620/938] [D loss: 1.3011] [G loss: 0.8340]
[Epoch 11/100] [Batch 182/938] [D loss: 1.3496] [G loss: 0.7699]
[Epoch 11/100] [Batch 682/938] [D loss: 1.2150] [G loss: 0.8919]
[Epoch 12/100] [Batch 244/938] [D loss: 1.2638] [G loss: 0.9146]
[Epoch 12/100] [Batch 744/938] [D loss: 1.3941] [G loss: 0.6635]
[Epoch 13/100] [Batch 306/938] [D loss: 1.3143] [G loss: 0.9099]
[Epoch 13/100] [Batch 806/938] [D loss: 1.3027] [G loss: 0.7418]
[Epoch 14/100] [Batch 368/938] [D loss: 1.3462] [G loss: 0.6388]
[Epoch 14/100] [Batch 868/938] [D loss: 1.2565] [G loss: 0.7786]
[Epoch 15/100] [Batch 430/938] [D loss: 1.2522] [G loss: 0.9937]
[Epoch 15/100] [Batch 930/938] [D loss: 1.2857] [G loss: 1.0986]
[Epoch 16/100] [Batch 492/938] [D loss: 1.3454] [G loss: 0.7573]
[Epoch 17/100] [Batch 54/938] [D loss: 1.3153] [G loss: 0.9889]
[Epoch 17/100] [Batch 554/938] [D loss: 1.2468] [G loss: 1.1739]
[Epoch 18/100] [Batch 116/938] [D loss: 1.4599] [G loss: 1.0137]
[Epoch 18/100] [Batch 616/938] [D loss: 1.2038] [G loss: 0.7575]
[Epoch 19/100] [Batch 178/938] [D loss: 1.3123] [G loss: 0.7688]
[Epoch 19/100] [Batch 678/938] [D loss: 1.2557] [G loss: 0.9564]
[Epoch 20/100] [Batch 240/938] [D loss: 1.2484] [G loss: 0.7339]
[Epoch 20/100] [Batch 740/938] [D loss: 1.2394] [G loss: 0.7991]
[Epoch 21/100] [Batch 302/938] [D loss: 1.2612] [G loss: 1.0142]
[Epoch 21/100] [Batch 802/938] [D loss: 1.2612] [G loss: 0.8492]
[Epoch 22/100] [Batch 364/938] [D loss: 1.2676] [G loss: 0.8447]
[Epoch 22/100] [Batch 864/938] [D loss: 1.2206] [G loss: 0.6595]
[Epoch 23/100] [Batch 426/938] [D loss: 1.2510] [G loss: 0.6739]
[Epoch 23/100] [Batch 926/938] [D loss: 1.3003] [G loss: 0.7826]
[Epoch 24/100] [Batch 488/938] [D loss: 1.1858] [G loss: 0.7580]
[Epoch 25/100] [Batch 50/938] [D loss: 1.2434] [G loss: 1.0509]
[Epoch 25/100] [Batch 550/938] [D loss: 1.3106] [G loss: 1.0460]
[Epoch 26/100] [Batch 112/938] [D loss: 1.3553] [G loss: 0.8753]
[Epoch 26/100] [Batch 612/938] [D loss: 1.3305] [G loss: 0.9353]
[Epoch 27/100] [Batch 174/938] [D loss: 1.2599] [G loss: 0.9273]
[Epoch 27/100] [Batch 674/938] [D loss: 1.3431] [G loss: 1.1888]
[Epoch 28/100] [Batch 236/938] [D loss: 1.2364] [G loss: 1.0220]
[Epoch 28/100] [Batch 736/938] [D loss: 1.3259] [G loss: 0.7538]
[Epoch 29/100] [Batch 298/938] [D loss: 1.2741] [G loss: 1.0393]
[Epoch 29/100] [Batch 798/938] [D loss: 1.3821] [G loss: 0.7309]
[Epoch 30/100] [Batch 360/938] [D loss: 1.2441] [G loss: 1.0607]
[Epoch 30/100] [Batch 860/938] [D loss: 1.2406] [G loss: 0.8472]
[Epoch 31/100] [Batch 422/938] [D loss: 1.2890] [G loss: 1.0351]
[Epoch 31/100] [Batch 922/938] [D loss: 1.1890] [G loss: 1.2030]
[Epoch 32/100] [Batch 484/938] [D loss: 1.2089] [G loss: 0.9127]
[Epoch 33/100] [Batch 46/938] [D loss: 1.2061] [G loss: 0.7472]
[Epoch 33/100] [Batch 546/938] [D loss: 1.2530] [G loss: 0.8877]
[Epoch 34/100] [Batch 108/938] [D loss: 1.2360] [G loss: 0.8802]
[Epoch 34/100] [Batch 608/938] [D loss: 1.3361] [G loss: 0.8364]
[Epoch 35/100] [Batch 170/938] [D loss: 1.3235] [G loss: 1.0804]
[Epoch 35/100] [Batch 670/938] [D loss: 1.2931] [G loss: 0.8127]
[Epoch 36/100] [Batch 232/938] [D loss: 1.1939] [G loss: 0.8350]
[Epoch 36/100] [Batch 732/938] [D loss: 1.2580] [G loss: 0.7571]
[Epoch 37/100] [Batch 294/938] [D loss: 1.2219] [G loss: 0.9296]
[Epoch 37/100] [Batch 794/938] [D loss: 1.2123] [G loss: 1.0306]
[Epoch 38/100] [Batch 356/938] [D loss: 1.2345] [G loss: 0.7284]
[Epoch 38/100] [Batch 856/938] [D loss: 1.2894] [G loss: 0.7053]
[Epoch 39/100] [Batch 418/938] [D loss: 1.1939] [G loss: 0.7684]
[Epoch 39/100] [Batch 918/938] [D loss: 1.2460] [G loss: 0.9547]
[Epoch 40/100] [Batch 480/938] [D loss: 1.2773] [G loss: 1.0253]
[Epoch 41/100] [Batch 42/938] [D loss: 1.1778] [G loss: 1.0214]
[Epoch 41/100] [Batch 542/938] [D loss: 1.2215] [G loss: 0.8861]
[Epoch 42/100] [Batch 104/938] [D loss: 1.2520] [G loss: 1.0879]
[Epoch 42/100] [Batch 604/938] [D loss: 1.1941] [G loss: 0.7307]
[Epoch 43/100] [Batch 166/938] [D loss: 1.2242] [G loss: 1.0486]
[Epoch 43/100] [Batch 666/938] [D loss: 1.2907] [G loss: 0.9880]
[Epoch 44/100] [Batch 228/938] [D loss: 1.0957] [G loss: 0.9254]
[Epoch 44/100] [Batch 728/938] [D loss: 1.1934] [G loss: 0.8492]
[Epoch 45/100] [Batch 290/938] [D loss: 1.2209] [G loss: 0.9815]
[Epoch 45/100] [Batch 790/938] [D loss: 1.3000] [G loss: 0.8677]
[Epoch 46/100] [Batch 352/938] [D loss: 1.2278] [G loss: 1.0309]
[Epoch 46/100] [Batch 852/938] [D loss: 1.2473] [G loss: 0.6890]
[Epoch 47/100] [Batch 414/938] [D loss: 1.1193] [G loss: 1.1348]
[Epoch 47/100] [Batch 914/938] [D loss: 1.2118] [G loss: 0.8892]
[Epoch 48/100] [Batch 476/938] [D loss: 1.2086] [G loss: 1.2849]
[Epoch 49/100] [Batch 38/938] [D loss: 1.1603] [G loss: 0.8094]
[Epoch 49/100] [Batch 538/938] [D loss: 1.1771] [G loss: 1.0013]
[Epoch 50/100] [Batch 100/938] [D loss: 1.2602] [G loss: 1.0529]
[Epoch 50/100] [Batch 600/938] [D loss: 1.2562] [G loss: 0.8363]
[Epoch 51/100] [Batch 162/938] [D loss: 1.2227] [G loss: 0.7376]
[Epoch 51/100] [Batch 662/938] [D loss: 1.1927] [G loss: 0.9985]
[Epoch 52/100] [Batch 224/938] [D loss: 1.3527] [G loss: 1.1608]
[Epoch 52/100] [Batch 724/938] [D loss: 1.2723] [G loss: 1.2899]
[Epoch 53/100] [Batch 286/938] [D loss: 1.2775] [G loss: 0.6443]
[Epoch 53/100] [Batch 786/938] [D loss: 1.2037] [G loss: 0.8052]
[Epoch 54/100] [Batch 348/938] [D loss: 1.1936] [G loss: 0.8085]
[Epoch 54/100] [Batch 848/938] [D loss: 1.2118] [G loss: 1.0519]
[Epoch 55/100] [Batch 410/938] [D loss: 1.2614] [G loss: 0.6429]
[Epoch 55/100] [Batch 910/938] [D loss: 1.2063] [G loss: 0.9023]
[Epoch 56/100] [Batch 472/938] [D loss: 1.1581] [G loss: 0.9893]
[Epoch 57/100] [Batch 34/938] [D loss: 1.2888] [G loss: 0.9309]
[Epoch 57/100] [Batch 534/938] [D loss: 1.2513] [G loss: 0.9157]
[Epoch 58/100] [Batch 96/938] [D loss: 1.1757] [G loss: 0.9943]
[Epoch 58/100] [Batch 596/938] [D loss: 1.2213] [G loss: 0.5816]
[Epoch 59/100] [Batch 158/938] [D loss: 1.2208] [G loss: 0.9956]
[Epoch 59/100] [Batch 658/938] [D loss: 1.2283] [G loss: 0.8126]
[Epoch 60/100] [Batch 220/938] [D loss: 1.2562] [G loss: 0.9837]
[Epoch 60/100] [Batch 720/938] [D loss: 1.1812] [G loss: 1.0592]
[Epoch 61/100] [Batch 282/938] [D loss: 1.2546] [G loss: 0.8844]
[Epoch 61/100] [Batch 782/938] [D loss: 1.1630] [G loss: 0.8407]
[Epoch 62/100] [Batch 344/938] [D loss: 1.2065] [G loss: 0.9433]
[Epoch 62/100] [Batch 844/938] [D loss: 1.3437] [G loss: 1.2230]
[Epoch 63/100] [Batch 406/938] [D loss: 1.1797] [G loss: 1.0741]
[Epoch 63/100] [Batch 906/938] [D loss: 1.2302] [G loss: 0.8402]
[Epoch 64/100] [Batch 468/938] [D loss: 1.1714] [G loss: 0.9637]
[Epoch 65/100] [Batch 30/938] [D loss: 1.2193] [G loss: 1.0411]
[Epoch 65/100] [Batch 530/938] [D loss: 1.1973] [G loss: 0.9927]
[Epoch 66/100] [Batch 92/938] [D loss: 1.1691] [G loss: 0.8396]
[Epoch 66/100] [Batch 592/938] [D loss: 1.1706] [G loss: 0.8079]
[Epoch 67/100] [Batch 154/938] [D loss: 1.1940] [G loss: 1.0591]
[Epoch 67/100] [Batch 654/938] [D loss: 1.2110] [G loss: 0.8448]
[Epoch 68/100] [Batch 216/938] [D loss: 1.3334] [G loss: 1.2425]
[Epoch 68/100] [Batch 716/938] [D loss: 1.1726] [G loss: 0.9447]
[Epoch 69/100] [Batch 278/938] [D loss: 1.1844] [G loss: 1.3587]
[Epoch 69/100] [Batch 778/938] [D loss: 1.1847] [G loss: 0.8058]
[Epoch 70/100] [Batch 340/938] [D loss: 1.1816] [G loss: 1.0102]
[Epoch 70/100] [Batch 840/938] [D loss: 1.0834] [G loss: 0.9057]
[Epoch 71/100] [Batch 402/938] [D loss: 1.1982] [G loss: 1.2689]
[Epoch 71/100] [Batch 902/938] [D loss: 1.2331] [G loss: 1.1821]
[Epoch 72/100] [Batch 464/938] [D loss: 1.1253] [G loss: 0.8630]
[Epoch 73/100] [Batch 26/938] [D loss: 1.1768] [G loss: 1.0590]
[Epoch 73/100] [Batch 526/938] [D loss: 1.2575] [G loss: 0.9302]
[Epoch 74/100] [Batch 88/938] [D loss: 1.1932] [G loss: 1.0069]
[Epoch 74/100] [Batch 588/938] [D loss: 1.2036] [G loss: 0.7066]
[Epoch 75/100] [Batch 150/938] [D loss: 1.1835] [G loss: 1.1273]
[Epoch 75/100] [Batch 650/938] [D loss: 1.1219] [G loss: 1.1903]
[Epoch 76/100] [Batch 212/938] [D loss: 1.2044] [G loss: 0.8930]
[Epoch 76/100] [Batch 712/938] [D loss: 1.2971] [G loss: 1.2523]
[Epoch 77/100] [Batch 274/938] [D loss: 1.1070] [G loss: 0.9315]
[Epoch 77/100] [Batch 774/938] [D loss: 1.1412] [G loss: 0.8089]
[Epoch 78/100] [Batch 336/938] [D loss: 1.1278] [G loss: 1.3260]
[Epoch 78/100] [Batch 836/938] [D loss: 1.1687] [G loss: 0.9051]
[Epoch 79/100] [Batch 398/938] [D loss: 1.1683] [G loss: 1.1012]
[Epoch 79/100] [Batch 898/938] [D loss: 1.2458] [G loss: 1.1635]
[Epoch 80/100] [Batch 460/938] [D loss: 1.1991] [G loss: 0.8094]
[Epoch 81/100] [Batch 22/938] [D loss: 1.3127] [G loss: 1.1219]
[Epoch 81/100] [Batch 522/938] [D loss: 1.2367] [G loss: 0.7774]
[Epoch 82/100] [Batch 84/938] [D loss: 1.1516] [G loss: 0.9732]
[Epoch 82/100] [Batch 584/938] [D loss: 1.1849] [G loss: 0.9335]
[Epoch 83/100] [Batch 146/938] [D loss: 1.1397] [G loss: 1.2962]
[Epoch 83/100] [Batch 646/938] [D loss: 1.2186] [G loss: 1.3127]
[Epoch 84/100] [Batch 208/938] [D loss: 1.1786] [G loss: 0.8326]
[Epoch 84/100] [Batch 708/938] [D loss: 1.0816] [G loss: 1.0449]
[Epoch 85/100] [Batch 270/938] [D loss: 1.2884] [G loss: 1.2043]
[Epoch 85/100] [Batch 770/938] [D loss: 1.3545] [G loss: 1.3558]
[Epoch 86/100] [Batch 332/938] [D loss: 1.1632] [G loss: 1.0463]
[Epoch 86/100] [Batch 832/938] [D loss: 1.2131] [G loss: 0.9421]
[Epoch 87/100] [Batch 394/938] [D loss: 1.4207] [G loss: 0.6200]
[Epoch 87/100] [Batch 894/938] [D loss: 1.2689] [G loss: 0.9638]
[Epoch 88/100] [Batch 456/938] [D loss: 1.1171] [G loss: 1.2163]
[Epoch 89/100] [Batch 18/938] [D loss: 1.0571] [G loss: 1.1640]
[Epoch 89/100] [Batch 518/938] [D loss: 1.1205] [G loss: 1.2385]
[Epoch 90/100] [Batch 80/938] [D loss: 1.1907] [G loss: 1.1529]
[Epoch 90/100] [Batch 580/938] [D loss: 1.2598] [G loss: 1.3407]
[Epoch 91/100] [Batch 142/938] [D loss: 1.0961] [G loss: 1.1675]
[Epoch 91/100] [Batch 642/938] [D loss: 1.1569] [G loss: 1.2329]
[Epoch 92/100] [Batch 204/938] [D loss: 1.1585] [G loss: 1.1988]
[Epoch 92/100] [Batch 704/938] [D loss: 1.2831] [G loss: 0.9801]
[Epoch 93/100] [Batch 266/938] [D loss: 0.9923] [G loss: 1.1239]
[Epoch 93/100] [Batch 766/938] [D loss: 1.1406] [G loss: 1.3908]
[Epoch 94/100] [Batch 328/938] [D loss: 1.1825] [G loss: 1.1348]
[Epoch 94/100] [Batch 828/938] [D loss: 1.0615] [G loss: 0.7669]
[Epoch 95/100] [Batch 390/938] [D loss: 1.1313] [G loss: 0.8031]
[Epoch 95/100] [Batch 890/938] [D loss: 1.1357] [G loss: 0.9063]
[Epoch 96/100] [Batch 452/938] [D loss: 1.0803] [G loss: 0.9468]
[Epoch 97/100] [Batch 14/938] [D loss: 1.1420] [G loss: 0.7302]
[Epoch 97/100] [Batch 514/938] [D loss: 1.1453] [G loss: 1.2820]
[Epoch 98/100] [Batch 76/938] [D loss: 1.1164] [G loss: 0.9781]
[Epoch 98/100] [Batch 576/938] [D loss: 1.1297] [G loss: 0.8417]
[Epoch 99/100] [Batch 138/938] [D loss: 1.0767] [G loss: 1.1055]
[Epoch 99/100] [Batch 638/938] [D loss: 1.1457] [G loss: 0.9654]
"""

def parse_training_output(output_text):
    """Parse training output to extract losses."""
    pattern = r'\[Epoch (\d+)/\d+\] \[Batch (\d+)/\d+\] \[D loss: ([\d.]+)\] \[G loss: ([\d.]+)\]'
    matches = re.findall(pattern, output_text)
    
    epochs = []
    batches = []
    d_losses = []
    g_losses = []
    iterations = []
    
    for match in matches:
        epoch, batch, d_loss, g_loss = match
        epoch = int(epoch)
        batch = int(batch)
        d_loss = float(d_loss)
        g_loss = float(g_loss)
        
        # Calculate iteration number (assuming 938 batches per epoch)
        iteration = epoch * 938 + batch
        
        epochs.append(epoch)
        batches.append(batch)
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        iterations.append(iteration)
    
    return np.array(epochs), np.array(batches), np.array(d_losses), np.array(g_losses), np.array(iterations)


def plot_gan_losses(epochs, batches, d_losses, g_losses, iterations, save_path='gan_training_losses.png'):
    """Create comprehensive loss plots."""
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Losses over iterations
    plt.subplot(2, 2, 1)
    plt.plot(iterations, d_losses, 'o-', label='Discriminator Loss', alpha=0.6, markersize=3)
    plt.plot(iterations, g_losses, 's-', label='Generator Loss', alpha=0.6, markersize=3)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('GAN Training Losses Over Iterations', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Losses over epochs (averaged)
    plt.subplot(2, 2, 2)
    unique_epochs = np.unique(epochs)
    d_loss_per_epoch = [d_losses[epochs == e].mean() for e in unique_epochs]
    g_loss_per_epoch = [g_losses[epochs == e].mean() for e in unique_epochs]
    
    plt.plot(unique_epochs, d_loss_per_epoch, 'o-', label='Discriminator Loss (avg)', linewidth=2, markersize=5)
    plt.plot(unique_epochs, g_loss_per_epoch, 's-', label='Generator Loss (avg)', linewidth=2, markersize=5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Average Loss Per Epoch', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Loss distribution
    plt.subplot(2, 2, 3)
    plt.hist(d_losses, bins=30, alpha=0.6, label='Discriminator Loss', color='blue', edgecolor='black')
    plt.hist(g_losses, bins=30, alpha=0.6, label='Generator Loss', color='orange', edgecolor='black')
    plt.xlabel('Loss Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Loss Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Loss statistics over time (rolling mean and std)
    plt.subplot(2, 2, 4)
    window = min(20, len(d_losses) // 5)  # Adaptive window size
    if window > 1:
        d_rolling_mean = np.convolve(d_losses, np.ones(window)/window, mode='valid')
        g_rolling_mean = np.convolve(g_losses, np.ones(window)/window, mode='valid')
        iterations_rolling = iterations[window-1:]
        
        plt.plot(iterations_rolling, d_rolling_mean, '-', label=f'D Loss (rolling mean, window={window})', linewidth=2)
        plt.plot(iterations_rolling, g_rolling_mean, '-', label=f'G Loss (rolling mean, window={window})', linewidth=2)
    else:
        plt.plot(iterations, d_losses, '-', label='Discriminator Loss', linewidth=2)
        plt.plot(iterations, g_losses, '-', label='Generator Loss', linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Smoothed Training Losses', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Loss plot saved to: {save_path}")
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("GAN Training Statistics")
    print("="*60)
    print(f"Total epochs: {len(unique_epochs)}")
    print(f"Total iterations logged: {len(iterations)}")
    print(f"\nDiscriminator Loss:")
    print(f"  Mean: {d_losses.mean():.4f}")
    print(f"  Std:  {d_losses.std():.4f}")
    print(f"  Min:  {d_losses.min():.4f}")
    print(f"  Max:  {d_losses.max():.4f}")
    print(f"\nGenerator Loss:")
    print(f"  Mean: {g_losses.mean():.4f}")
    print(f"  Std:  {g_losses.std():.4f}")
    print(f"  Min:  {g_losses.min():.4f}")
    print(f"  Max:  {g_losses.max():.4f}")
    print("="*60)


if __name__ == "__main__":
    print("Parsing training output...")
    epochs, batches, d_losses, g_losses, iterations = parse_training_output(training_output)
    
    print(f"Extracted {len(iterations)} training checkpoints")
    
    # Create plots
    plot_gan_losses(epochs, batches, d_losses, g_losses, iterations)
    
    # Save data for future use
    np.savez('parsed_loss_history.npz',
             epochs=epochs,
             batches=batches,
             d_losses=d_losses,
             g_losses=g_losses,
             iterations=iterations)
    print("\n✓ Loss data saved to: parsed_loss_history.npz")
