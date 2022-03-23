import pandas as pd
import matplotlib.pyplot as plt

weight_height = pd.read_csv('weight-height.csv')
weight_height= weight_height.drop(labels=[*range(1000, 10000)], axis=0)
weight_height.loc[(weight_height.Gender == "Male"), 'Gender'] = 1
weight_height.loc[(weight_height.Gender == "Female"), 'Gender'] = 0

wh = weight_height.sample(frac=1)
def weight_height_plot(dataset):
    dataset.plot(y='Height', kind='hist',
              color='red', title='Height (inch.) distribution')
    plt.show()

if __name__ == "__main__":
    weight_height_plot(wh)