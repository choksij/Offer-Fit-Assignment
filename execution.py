from __future__ import annotations
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt


class DataModeler:
    def __init__(self, sample_df: pd.DataFrame):                                                                     
        self._raw_train = sample_df.copy()                                  # initialization - raw training dataframe
        self.train_df: pd.DataFrame | None = None
        self._means: dict[str, float] = {}
        self._summary: str = ""


    def prepare_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        df = (self._raw_train if oos_df is None else oos_df).copy()                           # selection and conversion of features          
        df = df[['customer_id', 'amount', 'transaction_date']].set_index('customer_id')

        dt = pd.to_datetime(df['transaction_date'])
        df['transaction_date'] = dt.astype('int64').astype(float)
        df.loc[df['transaction_date'] < 0, 'transaction_date'] = np.nan           # mask missing dates

        if oos_df is None:
            self.train_df = df
        return df


    def impute_missing(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        df = self.train_df if oos_df is None else oos_df.copy()                # NaNs with training means (stored on first call)

        if oos_df is None:
            self._means = df.mean().to_dict()
        df = df.fillna(self._means)

        if oos_df is None:
            self.train_df = df
        return df


    def fit(self) -> None:
        if self.train_df is None:                                                     # summary threshold
            raise ValueError("Data not prepared. Call prepare_data() and impute_missing() first.")

        amt, date = self._means['amount'], self._means['transaction_date']
        self._summary = (
            f"Rule-based threshold model: predict True if amount ≥ {amt:.4f} "
            f"or transaction_date > {date:.0f}"
        )


    def model_summary(self) -> str:
        return self._summary                                     # fitted summary model


    def predict(self, oos_df: pd.DataFrame = None) -> np.ndarray:
        if oos_df is None:                                                 # producing predictions
            if self.train_df is None:
                raise ValueError("No data to predict on. Prepare data first.")
            df = self.train_df
        else:
            df = oos_df

        amt_thresh = self._means['amount']
        date_thresh = self._means['transaction_date']
        return ((df['amount'] >= amt_thresh) | (df['transaction_date'] > date_thresh)).values


    def save(self, path: str) -> None:
        with open(path, "wb") as f:                                     # serailization
            pickle.dump(self, f)


    def plot_training_data(self) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd

        df = self.train_df
        dates = pd.to_datetime(df['transaction_date'], unit='ns')
        amounts = df['amount']

        plt.figure()
        plt.scatter(dates, amounts, s=50)
        plt.axhline(self._means['amount'], linestyle='--', label='amount threshold')
        date_thr = pd.to_datetime(self._means['transaction_date'], unit='ns')
        plt.axvline(date_thr, linestyle='--', label='date threshold')

        plt.title('Training data with decision thresholds')
        plt.xlabel('transaction_date')
        plt.ylabel('amount')
        plt.legend()

        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()

        plt.savefig("training_scatter.png", dpi=150)
        plt.show()


    def plot_feature_distributions(self) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd
        plt.figure()
        plt.hist(self.train_df['amount'], bins=10)
        plt.title('Amount Distribution')
        plt.xlabel('amount')
        plt.ylabel('frequency')
        plt.savefig("amount_histogram.png", dpi=150)
        plt.show()

        dates = pd.to_datetime(self.train_df['transaction_date'], unit='ns')
        plt.figure()
        plt.hist(dates, bins=10)
        plt.title('Transaction Date Distribution')
        plt.xlabel('transaction_date')
        plt.ylabel('frequency')

        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()

        plt.savefig("date_histogram.png", dpi=150)
        plt.show()


    @staticmethod
    def load(path: str) -> DataModeler:
        with open(path, "rb") as f:                                     # loading datamodeler
            return pickle.load(f)


transact_train_sample = pd.DataFrame({
    "customer_id": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "amount": [1, 3, 12, 6, 0.5, 0.2, np.nan, 5, np.nan, 3],
    "transaction_date": [
        '2022-01-01', '2022-08-01', None, '2022-12-01',
        '2022-02-01', None, '2022-02-01', '2022-01-01',
        '2022-11-01', '2022-01-01'
    ],
    "outcome": [False, True, True, True, False, False, True, True, True, False]
})

print(f"Training sample:\n{transact_train_sample}\n")
print(f"Current dtypes:\n{transact_train_sample.dtypes}\n")

dm = DataModeler(transact_train_sample)
dm.prepare_data()
print(f"Changed columns to:\n{dm.train_df.dtypes}\n")
dm.impute_missing()
print(f"Imputed missing as mean:\n{dm.train_df}\n")

print("Fitting model")
dm.fit()
print(f"Fit model:\n{dm.model_summary()}\n")

in_sample_preds = dm.predict()
print(f"Predicted on training sample: {in_sample_preds}\n")
print(f"Accuracy = {sum(in_sample_preds == [False, True, True, True, False, False, True, True, True, False])/.1}%\n")

dm.save("transact_modeler")
loaded = DataModeler.load("transact_modeler")
print(f"Loaded model summary:\n{loaded.model_summary()}\n")

transact_test_sample = pd.DataFrame({
    "customer_id": [21, 22, 23, 24, 25],
    "amount": [0.5, np.nan, 8, 3, 2],
    "transaction_date": ['2022-02-01', '2022-11-01', '2022-06-01', None, '2022-02-01']
})

test_prepared = dm.prepare_data(transact_test_sample)
print(f"Changed columns to:\n{test_prepared.dtypes}\n")
test_filled = dm.impute_missing(test_prepared)
print(f"Imputed missing as mean:\n{test_filled}\n")

oos_preds = dm.predict(test_filled)
print(f"Predicted on out of sample data: {oos_preds}\n")
print(f"Accuracy = {sum(oos_preds == [False, True, True, False, False])/.05}%")


dm.plot_training_data()
dm.plot_feature_distributions()
