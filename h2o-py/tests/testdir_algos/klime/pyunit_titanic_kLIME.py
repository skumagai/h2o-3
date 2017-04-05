from __future__ import print_function
from builtins import range
import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.klime import H2OKlimeEstimator



def kLIMEtitanic():
    # Load Titanic dataset (with predictions of 'Survived' made by GBM)
    titanic_data = h2o.import_file(path = pyunit_utils.locate("smalldata/klime_test/titanic_preds.csv"),
                                   col_types=["enum","enum","real","real","real","enum","real","real","real","real"])
    titanic_data.describe()

    # Train a k-LIME model
    titanic_klime = H2OKlimeEstimator(seed=12345, k=12)
    titanic_klime.train(training_frame=titanic_data, y="p1",
                        ignored_columns=["PassengerId", "Survived", "predict", "p0"])
    titanic_klime.show()

    # Use as a regular regression model to predict
    p1_predicted = titanic_klime.predict(titanic_data)
    p1_predicted.show()

    model = h2o.get_model(titanic_klime.model_id)
    model.show()

    mse_manual = ((titanic_data["p1"] - p1_predicted) * (titanic_data["p1"] - p1_predicted)).mean()[0,0]

    assert abs(model.mse() - 0.0026993720936) < 1e-6
    assert abs(model.mse() - mse_manual) < 1e-6

if __name__ == "__main__":
    pyunit_utils.standalone_test(kLIMEtitanic)
else:
    kLIMEtitanic()
