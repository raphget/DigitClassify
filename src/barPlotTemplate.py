# Print a bar chart with groups

import numpy as np
import matplotlib.pyplot as plt

# Plots main classification metrics for a given model
def generateBarPlot(report, title, outfile):
    precision = []
    recall = []
    f1 = []
    #pprint.pprint(report)
    for x in range(0,10):
            precision.append(float(report[str(x)]['precision']))
            recall.append(float(report[str(x)]['recall']))
            f1.append(float(report[str(x)]['f1-score']))

    barPlotTemplate(precision,recall,f1, title, outfile)

def barPlotTemplate(bars1,bars2,bars3, title, outfile):

    barWidth = 0.25
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    # Make the plot
    plt.grid(axis='y',zorder = 0)
    plt.bar(r1, bars1, color='blue', width=barWidth, edgecolor='white',zorder = 3, label = 'Precision')
    plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', zorder = 3, label = 'Recall')
    plt.bar(r3, bars3, color='black', width=barWidth, edgecolor='white', zorder = 3, label = 'F1-score')
    
    plt.title(title)
    
    plt.xlabel('True Class', fontweight='bold')
    plt.ylabel('Precision')
    plt.yticks(np.arange(0,1,0.05))
    plt.xticks([r + barWidth for r in range(len(bars1))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.legend()
    #plt.show()
    plt.savefig(outfile,dpi=400,bbox_inches='tight',pad_inches=0.05) # save as a pdf
    plt.clf()