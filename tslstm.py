import os
import numpy
import pandas
import logging
import seaborn
import warnings

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Flatten, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# logging config
logging.basicConfig(
    filename="logs.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
    )

class TSLSTM(object):
    """ Bayes By Backprop """

    def __init__(self, path_to_csv, inputs:list, neurons:int, m:int, n:int, approach:str):
        """
        Arguments:
        ---------
        path_to_csv : str
            absolute path to the csv file
        inputs : list
            list of column indices that are needed as input
            it should include 0 as the "Date" column
            it should include index for "Output" column
        neurons : int
            number of hidden units/neuron per LSTM
        m : int
            moving window of ‘m’ number of time step inputs
        n : int
            ‘n’ day output multistep forecasting
        approach : string
            if "lstm" then a model built using LSTM is returned
            if "lstmd" the a model built using LSTM followed by dropout is
            returned
        """
        self._path_to_csv = path_to_csv
        self._inputs = inputs
        self._neurons = neurons
        self._m = m
        self._n = n
        self._approach = approach

    def load_dataset(self, get_overview:bool):
        """
        To load a csv file that should to have a 'Date' column at the 0th index.
        Date values are sorted in ascending order (oldest first) with dayfirst.
        Arguments:
        ----------
        get_overview : boolean
            if overview of the dataset is required
        Returns:
        --------
        data : dataframe
            dataframe with 'Date' column converted to date-time and set as index
        """
        
        self.data = pandas.read_csv(
            self._path_to_csv,
            usecols = self._inputs
            )
        self.data['Date'] = pandas.to_datetime(self.data['Date'], dayfirst=True)
        self.data.sort_values(by=['Date'], inplace=True, ascending=True)
    
        if get_overview:
            self._data_overview()

        return self.data

    def _data_overview(self):
        """
        To get general information about the dataset which is stroed in log file
        """
        # general description of data features
        logging.info(self.data.describe())

        # count null values present
        if self.data.isnull().all().sum() == 0:
            logging.info("No null values present")
        else:
            logging.info("{} null values present".format(
                self.data.isnull().all().sum())
                )
        
        # total number of data points and features
        logging.info("{} data points with {} features were found".format(
            self.data.shape[0],
            self.data.shape[1]
        ))

        # total unique date values
        logging.info("{} unique date values were found".format(
            self.data.index.nunique()
        ))

        # oldest and latest date value
        logging.info("The oldest date in dataset is {}".format(
            self.data.index.min()
        ))
        logging.info("The latest date in dataset is {}".format(
            self.data.index.max()
        ))

        # visualize the data
        if not os.path.isdir("./images"):
            os.makedirs("./images")
        fig = self.data.plot(
            x='Date',
            y=list(self.data)[1:],
            style="-",
            alpha=0.7,
            figsize=(20, 10),
            xlabel = 'Date',
            ylabel='Feature Values'
            ).get_figure()
        fig.savefig("images/initial-plot.pdf")

    def _split_sequences(self, sequences):
        """
        Split a multivariate sequence (dataset) into samples X and y
        Arguments:
        ----------
        sequences : Dataframe
            train and test dataframe with the features and target column
        Returns:
        -------
        X, y : numpy arrays
            sequence split into features and target
        """
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + self._m
            out_end_ix = end_ix + self._n-1
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x = sequences[i:end_ix, :-1]
            seq_y = sequences[end_ix-1:out_end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)

        return numpy.array(X), numpy.array(y)

    def _standardscaler_transform(self, train, test):
        """
        Standard scale the train and test data passed
        Removes the date column from consideration
        Arguments:
        ---------
        train, test : pandas dataframe
            Train and test split of the dataset
        Returns:
        -------
        train_data, test_date : numpy arrays
            Standard scaled train and test dataframes
        """
        X_train = numpy.array(train.iloc[:,1:-1])
        y_train = numpy.array(train.iloc[:, -1])
        X_test = numpy.array(test.iloc[:,1:-1])
        y_test = numpy.array(test.iloc[:,-1])
        
        X_train = self.scalerx.fit_transform(X_train)
        y_train = self.scalery.fit_transform(y_train.reshape(-1,1))

        X_test = self.scalerx.transform(X_test)
        y_test = self.scalery.transform(y_test.reshape(-1,1))
        
        train_data = numpy.hstack((X_train, y_train))
        test_data = numpy.hstack((X_test, y_test))
        
        return train_data, test_data

    def _build_model(self):
        """
        build 2x LSTM model followed by 1 FC with MSE as the loss function
        Returns:
        -------
        model : keras.engine.sequential.Sequential object
            compiled model
        """
        if self._approach == "lstm":
            logging.info(
                "Building model using LSTM . . ."
                )
            model = Sequential()
            model.add(
                LSTM(
                    self._neurons,
                    activation='relu',
                    return_sequences=True,
                    input_shape=(self._m, self.n_features)
                    )
                )
            model.add(LSTM(self._neurons, activation='relu'))
            model.add(Flatten())
            model.add(Dense(self._n))
            model.compile(optimizer='adam', loss='mse')

        elif self._approach == "lstmd":
            logging.info(
                "Building model using LSTM followed by Dropout . . ."
                )
            model = Sequential()
            model.add(
                LSTM(
                    self._neurons,
                    activation='relu',
                    return_sequences=True,
                    input_shape=(self._m, self.n_features)
                    )
                )
            model.add(LSTM(self._neurons, activation='relu'))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(self._n))
            model.compile(optimizer='adam', loss='mse')

        else:
            logging.info("Please use either of 'lstm' or 'lstmd' as the input")
            logging.info("Terminating program . . . ")
            exit()

        return model

    def scale_split_train(self):
        """
        Use timeseries split and perform standard scaling on the data
        """
        self.scalerx, self.scalery = StandardScaler(), StandardScaler()
        tscv = TimeSeriesSplit(n_splits=5)
        learning_rate_reduction = ReduceLROnPlateau(
            monitor = 'val_loss', 
            patience = 2, 
            verbose = 1, 
            factor = 0.5, 
            min_lr = 0.0001,
            mode = "auto",
            min_delta = 0.0001,
            cooldown = 5
            )
        earlyStop = EarlyStopping(
            monitor = "val_loss",
            verbose = 2,
            mode = 'auto',
            patience = 20
            )
        i = 1

        for train_index, test_index in tscv.split(self.data):
            logging.info("Starting with split {} out of 5".format(i))
            # splitting the dataset into train and test
            self.train = self.data.iloc[train_index,:]
            self.test = self.data.iloc[test_index,:]
     
            # standardscaling the train and test data
            train_data, test_data = self._standardscaler_transform(
                self.train,
                self.test
                )
  
            # covert into input/output for train data
            X_train, y_train = self._split_sequences(train_data)
            # covert into input/output for test data
            X_test, y_test = self._split_sequences(test_data)
            # the dataset knows the number of features, e.g. 2
            self.n_features = X_train.shape[2]

            # define model
            self.model = self._build_model()
            logging.info("Model Summary: {}".format(
                self.model.summary(
                    print_fn=logging.info
                    )
                )
            )
            # fitting the model
            logging.info("Training started . . .")
            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data = (X_test, y_test),
                epochs = 1000,
                verbose = 2,
                callbacks = [learning_rate_reduction, earlyStop]
                )
            logging.info("Training Logs: {}".format(self.history.history))
            logging.info("Training Completed")

            if i == 5:
                predictions = self.model.predict(X_test)
                self.predictions = self.scalery.inverse_transform(predictions)
                
            i += 1   
                
    def _reading_predictions(self, predictions):
        """
        Convert predictions to arrays with values for "m-1" rows "0"
        Returns:
        -------
        transformed_predictions : numpy array
            predictions converted to required format
        """
        transformed_predictions=[]
        for _ in range(self._m-1):
            transformed_predictions.append(0)

        for i in range(self._m-1, predictions.shape[0]+self._n):
            transformed_predictions.append(predictions[i-self._m+1][0])

            if i-self._m+1 == predictions.shape[0]-1:
                for j in range(self._n-1):
                    transformed_predictions.append(predictions[i-self._m+1][j+1])

        return numpy.array(transformed_predictions)    

    def _dataframe_predictions(self):
        """
        Joining the transformed predictions with test index and date
        Returns:
        -------
        predictions_df : pandas dataframe
            transformed predictions converted to dataframe as required
        """

        df_dict = {
            "Index": self.test.index,
            "Date" : self.test.Date,
            "Output": self._reading_predictions(self.predictions)
        }
        
        predictions_df = pandas.DataFrame(
            data = df_dict,
            index = df_dict["Index"],
            columns = ["Date", "Output"]
        )
  
        return predictions_df

    def plot_dataframe(self):
        """
        Prepare the dataframe that needs to be plotted with labels as history,
        prediction and true
        Plot and save the figure
        Returns:
        -------
        plot_df : pandas dataframe
            dataframe that needs to be plotted
        """
        logging.info("Preparing the dataframe for plotting . . .")
        predictions_df = self._dataframe_predictions()
        plot_test = pandas.DataFrame(
            index = self.test.index, 
            columns = ["Date","Output", "Type"]
            )
        start_index = plot_test.index[0]

        for rows in plot_test.itertuples():

            plot_test.loc[rows.Index, "Date"] = self.data.loc[
                rows.Index, 
                "Date"
                ]
            if rows.Index < start_index + self._m - 1:
                plot_test.loc[rows.Index,"Output"] = self.data.loc[
                    rows.Index, 
                    "Output"
                    ]
                plot_test.loc[rows.Index, "Type"] = "History"
            else:
                plot_test.loc[rows.Index,"Output"] = predictions_df.loc[
                    rows.Index,
                    "Output"
                    ]
                plot_test.loc[rows.Index, "Type"] = "Predicted"

        plot_train = pandas.DataFrame(
            index = self.train.index, 
            columns = ["Date","Output", "Type"])

        for rows in plot_train.itertuples():

            plot_train.loc[rows.Index, "Date"] = self.data.loc[
                rows.Index, 
                "Date"
                ]
            plot_train.loc[rows.Index,"Output"] = self.data.loc[
                rows.Index, 
                "Output"
                ]
            plot_train.loc[rows.Index, "Type"] = "History"
            
        plot_df = pandas.concat([plot_train,plot_test], axis=0)
        plot_df.Output = plot_df.Output.apply(pandas.to_numeric)
        logging.info("Plotting DataFrame: {}".format(plot_df))

        y_std = plot_df.Output.std()
        error1 = y_std
        error2 = 1.96*y_std
   
        under_line_1 = predictions_df.Output[self._m-1:] - error1
        over_line_1 = predictions_df.Output[self._m-1:] + error1

        under_line_2 = predictions_df.Output[self._m-1:] - error2
        over_line_2 = predictions_df.Output[self._m-1:] + error2

        plt.figure(figsize=(20,10))
        g = seaborn.lineplot(
            data = plot_df,
            x = "Date",
            y = "Output",
            hue = "Type",
            legend = False
            )
        g = seaborn.lineplot(
            x = plot_df.Date,
            y = self.test.Output[self._m-2:],
            color = "green",
            alpha=0.6
            )
        plt.fill_between(
            predictions_df.Date[self._m-1:],
            under_line_1,
            over_line_1,
            color='g',
            alpha=.1
            )
        plt.fill_between(
            predictions_df.Date[self._m-1:],
            under_line_2,
            over_line_2,
            color='g',
            alpha=.05
            )

        # title
        new_title = 'Model Output'
        g.set_title(new_title)
        # replace labels
        plt.legend(
            title = 'Keys',
            loc = 'upper left',
            labels = ['History', 'Predicted', "True"]
            )
        plt.savefig("./images/model-output.pdf")

        logging.info("Plot saved under images directory")

        return plot_df, plot_test, plot_train  

    def main(self):
        self.load_dataset(get_overview=True)
        self.scale_split_train()
        self.plot_dataframe()

if __name__ == "__main__":
    obj = TSLSTM(
        path_to_csv="./data.csv", 
        inputs=[0,1,2,3,4,5], 
        neurons=50, 
        m=2, 
        n=1,
        approach="lstmd"
        )
    obj.main()