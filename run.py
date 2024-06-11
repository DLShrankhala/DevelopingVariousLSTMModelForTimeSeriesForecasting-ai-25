from Core.data import Data
from Core.config import Column
from Core.model import LSTM_Trainer

def main():
    data = Data()
    data.read('Data/AAPL.csv')
    data.check_null_values()
    data.clean_data()
    print(Column.OPEN.value)
    data.print_head()
    data.print_description()
    scaled_data, scaler = data.normalize()
    data.visualize(Column.OPEN.value)
    data.visualize(Column.CLOSE.value)

    train_data, val_data, test_data = data.split_data(scaled_data)
    x_train, y_train = data.prepare_data(train_data)
    x_val, y_val = data.prepare_data(val_data)
    x_test, y_test = data.prepare_data(test_data)

    trainer = LSTM_Trainer(x_train, y_train, x_val, y_val, x_test, y_test, scaler)
    trainer.build_and_train_lstm()
    trainer.predict_and_plot()
    trainer.evaluate_model()

if __name__ == "__main__":
    main()
