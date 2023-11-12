import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np


def tukeyplot(results, dim=None, yticklabels=None):
    npairs = len(results)

    fig = plt.figure()
    fsp = fig.add_subplot(111)
    fsp.axis([-50,50,0.5,10.5])
    fsp.set_title('95 % family-wise confidence level')
    fsp.title.set_y(1.025)
    fsp.set_yticks(np.arange(1,11))
    fsp.set_yticklabels(['V-T','V-S','T-S','V-P','T-P','S-P','V-M',
                         'T-M','S-M','P-M'])
    #fsp.yaxis.set_major_locator(mticker.MaxNLocator(npairs))
    fsp.yaxis.grid(True, linestyle='-', color='gray')
    fsp.set_xlabel('Differences in mean levels of Var', labelpad=8)
    fsp.xaxis.tick_bottom()
    fsp.yaxis.tick_left()

    xticklines = fsp.get_xticklines()
    for xtickline in xticklines:
        xtickline.set_marker(lines.TICKDOWN)
        xtickline.set_markersize(10)

    xlabels = fsp.get_xticklabels()
    for xlabel in xlabels:
        xlabel.set_y(-.04)

    yticklines = fsp.get_yticklines()
    for ytickline in yticklines:
        ytickline.set_marker(lines.TICKLEFT)
        ytickline.set_markersize(10)

    ylabels = fsp.get_yticklabels()
    for ylabel in ylabels:
        ylabel.set_x(-.04)

    for pair in range(npairs):
        data = .5+results[pair]/100.
        #fsp.axhline(y=npairs-pair, xmin=data[0], xmax=data[1], linewidth=1.25,
        fsp.axhline(y=npairs-pair, xmin=data.mean(), xmax=data[1], linewidth=1.25,
            color='blue', marker="|",  markevery=1)

        fsp.axhline(y=npairs-pair, xmin=data[0], xmax=data.mean(), linewidth=1.25,
            color='blue', marker="|", markevery=1)

    #for pair in range(npairs):
    #    data = .5+results[pair]/100.
    #    data = results[pair]
    #    data = np.r_[data[0],data.mean(),data[1]]
    #    l = plt.plot(data, [npairs-pair]*len(data), color='black',
    #                linewidth=.5, marker="|", markevery=1)

    fsp.axvline(x=0, linestyle="--", color='black')

    fig.subplots_adjust(bottom=.125)



results = np.array([[-10.04391794,  26.34391794],
      [-21.45225794,  14.93557794],
      [  5.61441206,  42.00224794],
      [-13.40225794,  22.98557794],
      [-29.60225794,   6.78557794],
      [ -2.53558794,  33.85224794],
      [-21.55225794,  14.83557794],
      [  8.87275206,  45.26058794],
      [-10.14391794,  26.24391794],
      [-37.21058794,  -0.82275206]])


#plt.show()
