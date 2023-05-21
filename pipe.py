from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

# test = pd.DataFrame([25,'Private',7,'Never-married','Machine-op-inspct','Own-child','Black','Male',40]).transpose()
df = pd.read_csv('./Data/adult.csv')
df.drop_duplicates(inplace=True)
df.drop(columns=['fnlwgt', 'capital-gain', 'capital-loss',
        'native-country', 'education'], axis=1, inplace=True)
df.replace('?', 'other', inplace=True)
df.rename(columns={'age': 0, 'workclass': 1, 'educational-num': 2, 'marital-status': 3,
          'occupation': 4, 'relationship': 5, 'race': 6, 'gender': 7, 'hours-per-week': 8}, inplace=True)
le = LabelEncoder()
df['income'] = le.fit_transform(df['income'])
trf1 = ColumnTransformer(
    [
        ('normalization', MinMaxScaler(), [0, 8])
    ], remainder='passthrough'
)
trf2 = ColumnTransformer(
    [
        ('encoding', OneHotEncoder(sparse_output=False), [2, 4, 5, 6, 7, 8])
    ], remainder='passthrough'
)


def clean(obj):
    obj = pd.DataFrame(obj)
    return obj[[11,  46, 44, 45, 43, 19, 25,  4, 21, 16, 33, 39,  9, 35, 23,  34]]


ft = FunctionTransformer(clean)

pipe = Pipeline(
    [
        ('Scaling Data', trf1),
        ('Encoding Data', trf2),
        ('data', ft),
        ('classifier', MultinomialNB())
    ]
)

pipe.fit(df.iloc[:, :9].copy(), df['income'].copy())

def predict(data):
    return pipe.predict(data)

# print(pipe.predict(test))
