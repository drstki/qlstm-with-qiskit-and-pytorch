import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

class DATAPREP: # load and prepare data set for using quantum backend

    print('________________________________________________________________________')
    print()
    print("Load and Preprocessing DATA:")
    print()

    def prep_stripes(dataset,sep, header, x_axis, y_axis, idco, FIRST, LAST, FIRST_REFERENCE, LAST_REFERENCE, figsize, LIM, LAST_TRAIN, LAST_VALID, LAST_TEST, periods, dataset_curve_file):
        
        FIRST = 1850
        LAST = 2018  # inclusive

        # Reference period for the center of the color scale

        FIRST_REFERENCE = 1971
        LAST_REFERENCE = 2000
        LIM = 0.7 # degrees

        data_file = pd.read_csv(dataset, sep=sep, header=header, names=[x_axis, y_axis], index_col=[idco])
        
        anomaly = data_file.loc[FIRST:LAST, y_axis].dropna()
        
        reference = anomaly.loc[FIRST_REFERENCE:LAST_REFERENCE].mean()
    
        cmap = ListedColormap(
            ['#08306b', '#08519c', '#2171b5', '#4292c6','#6baed6', 
             '#9ecae1', '#c6dbef', '#deebf7','#fee0d2', '#fcbba1', 
             '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
             )

        # print and save data image 
        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        # create a collection with a rectangle for each year

        col = PatchCollection([Rectangle((y, 0), 1, 1) for y in range(FIRST, LAST + 1)])

        # set data, colormap and color limits

        col.set_array(anomaly)
        col.set_cmap(cmap)
        col.set_clim(reference - LIM, reference + LIM)
        ax.add_collection(col)

        ax.set_ylim(0, 1)
        ax.set_xlim(FIRST, LAST + 1)
    
        fig.set_title('Warming Stripes 1850-2018')
        fig = plt.savefig('./results/qlstm_warming_stripes_data.png')

        train_df = data_file.loc[FIRST:LAST_TRAIN, y_axis].dropna().reset_index().set_index(x_axis)
        valid_df = data_file.loc[LAST_TRAIN+1:LAST_VALID, y_axis].dropna().reset_index().set_index(x_axis)
        test_df = data_file.loc[LAST_VALID+1:LAST_TEST, y_axis].dropna().reset_index().set_index(x_axis)

        train_df = pd.concat([train_df,train_df[y_axis].shift(periods=periods)],axis=1)
        valid_df = pd.concat([valid_df,valid_df[y_axis].shift(periods=periods)],axis=1)
        test_df = pd.concat([test_df,test_df[y_axis].shift(periods=periods)],axis=1)
    
        train_scaler_amount=MinMaxScaler()
        transform_1 = y_axis
        transform_2 = '%s_periods[%d]' % (y_axis, 0)
        transform_3 = '%s_periods[%d]' % (y_axis, 1)
       
        #train_df = train_scaler_amount.fit_transform(train_df[[y_axis,'%s_periods[%d]' % (y_axis, periods[0]),'%s_periods[%d]' % (y_axis, periods[1])]])[periods[1]:,:]
        #valid_df = train_scaler_amount.transform(valid_df[[y_axis,'%s_periods[%d]' % (y_axis, periods[0]),'%s_periods[%d]' % (y_axis, periods[1])]])[periods[1]:,:]
        #test_df = train_scaler_amount.transform(test_df[[y_axis,'%s_periods[%d]' % (y_axis, periods[0]),'%s_periods[%d]' % (y_axis, periods[1])]])[periods[1]:,:]
        train_df=train_scaler_amount.fit_transform(train_df[['anomaly','%s_%d' % (y_axis, periods[0]),'%s_%d' % (y_axis, periods[1])]])[5:,:]
        valid_df=train_scaler_amount.transform(valid_df[['anomaly','%s_%d' % (y_axis, periods[0]),'%s_%d' % (y_axis, periods[1])]])[5:,:]
        test_df=train_scaler_amount.transform(test_df[['anomaly','%s_%d' % (y_axis, periods[0]),'%s_%d' % (y_axis, periods[1])]])[5:,:]
    
        plt.figure(figsize=figsize)
        plt.plot(train_df[:,0], color='blue', label='Train data')
        fig = plt.savefig(dataset_curve_file)

        return train_df, valid_df, test_df
    
