from data import Data
from config import Column
from model import LSTM_Trainer

data = Data()
data.read('Data/BAJAJ-AUTO.csv')
data.check_null_values()
data.clean_data()
print(Column.OPEN.value)
data.print_head()
data.print_description()
data.normalize()
data.visualize(Column.OPEN.value)
data.visualize(Column.CLOSE.value)

trainer = LSTM_Trainer(data.dataframe, data.scaler)
trainer.build_and_train_lstm()
trainer.predict_and_plot()
trainer.evaluate_model()
trainer.save_model('PreTrainedModel/BAJAj-AUTO_lstm_model.h5')
trainer.load_model('PreTrainedModel/BAJAj-AUTO_lstm_model.h5')
