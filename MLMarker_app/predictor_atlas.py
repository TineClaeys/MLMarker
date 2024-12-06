import pandas as pd
import random
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

class PredictorAtlas:
    def __init__(self, atlas_path):
        # Initialize the atlas from a CSV file
        self.atlas = pd.read_csv(atlas_path, sep=',')
        
    def testje(self):
        # Test method to print the atlas type
        print('hallokidoki')
        print(type(self.atlas))

    def balance_atlas(self):
        # Balance the atlas by dropping low abundant tissues and undersampling the majority class
        tissue_counts = self.atlas['tissue_name'].value_counts().to_frame()
        low_tissues = tissue_counts.index[tissue_counts['tissue_name'] < 3].tolist()
        self.atlas = self.atlas[~self.atlas['tissue_name'].isin(low_tissues)]
        tf = dict(Counter(self.atlas['tissue_name']))
        tf = sorted(tf.items(), key=lambda item: item[1], reverse=True)
        tf = dict(tf)
        first_tissue, first_value = list(tf.items())[0]
        second_tissue, second_value = list(tf.items())[1]
        tf[first_tissue] = second_value
        undersample = RandomUnderSampler(sampling_strategy=tf)
        y = self.atlas.pop('tissue_name').to_numpy()
        X = self.atlas
        X_under, y_under = undersample.fit_resample(X, y)
        self.atlas_under = X_under
        self.atlas_under['tissue_name'] = y_under
        return self.atlas_under, tf

    def train_test_split(self, ratio):
        # Split the atlas into training and testing datasets
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        tissues = self.atlas['tissue_name'].unique()
        DataFrameDict = {elem: pd.DataFrame for elem in tissues}
        for key in DataFrameDict.keys():
            DataFrameDict[key] = self.atlas[self.atlas['tissue_name'] == key]
        for key in DataFrameDict.keys():
            train = self.sampleData(DataFrameDict[key], ratio)
            train_df = train_df.append(train)
            test = DataFrameDict[key].drop(train.index)
            test_df = test_df.append(test)
        self.y_train = train_df.pop('tissue_name').values
        X_train = train_df.values
        self.y_test = test_df.pop('tissue_name').values
        X_test = test_df.values
        self.X_train = pd.DataFrame(X_train, columns=self.atlas.columns)
        self.X_test = pd.DataFrame(X_test, columns=self.atlas.columns)
        return self.X_train, self.y_train, self.X_test, self.y_test

    def class_weights(self, tf):
        # Calculate class weights based on the frequency of each class in the balanced atlas
        som = self.atlas_under.shape[0]
        weight_and_label = pd.DataFrame(columns=['label', 'weight'])
        for i, (key, value) in enumerate(tf.items()):
            w = (som - value) / value
            weight_and_label.loc[i] = [key, w]
        train_label_weight = pd.merge(pd.DataFrame(self.y_train, columns=['label']), weight_and_label, how='left', on='label')
        train_weights = train_label_weight['weight'].to_numpy().flatten()
        self.dict_train_label_weights = dict(zip(train_label_weight.label, train_label_weight.weight))
        return self.dict_train_label_weights, train_weights

    def sampleData(self, df, ratio):
        # Sample a subset of the data based on the given ratio
        df_size = len(df.index)
        sample_size = max(1, min(int(round(ratio * df_size)), df_size - 1))
        indexes = random.sample(range(df_size), sample_size)
        sample = df.iloc[indexes]
        return sample

    def cv_comparison(self, models, names, X, y, cv):
        # Perform cross-validation comparison of different models
        cv_scores = pd.DataFrame()
        scores = {
            'Accuracy': [], 'f1_macro': [], 'precision_macro': [], 'recall_macro': [],
            'f1_weighted': [], 'precision_weighted': [], 'recall_weighted': []
        }
        for model, name in zip(models, names):
            print(name)
            start = time.time()
            acc = np.round(cross_val_score(model, X, y, scoring='accuracy', cv=cv), 4)
            f1 = np.round(cross_val_score(model, X, y, scoring='f1_macro', cv=cv), 4)
            prec = np.round(cross_val_score(model, X, y, scoring='precision_macro', cv=cv), 4)
            rec = np.round(cross_val_score(model, X, y, scoring='recall_macro', cv=cv), 4)
            f1_w = np.round(cross_val_score(model, X, y, scoring='f1_weighted', cv=cv), 4)
            prec_w = np.round(cross_val_score(model, X, y, scoring='precision_weighted', cv=cv), 4)
            rec_w = np.round(cross_val_score(model, X, y, scoring='recall_weighted', cv=cv), 4)
            cv_scores[name] = [
                round(np.mean(acc[~np.isnan(acc)]), 4), round(np.mean(f1[~np.isnan(f1)]), 4),
                round(np.mean(prec[~np.isnan(prec)]), 4), round(np.mean(rec[~np.isnan(rec)]), 4),
                round(np.mean(f1_w[~np.isnan(f1_w)]), 4), round(np.mean(prec_w[~np.isnan(prec_w)]), 4),
                round(np.mean(rec_w[~np.isnan(rec_w)]), 4)
            ]
            print(f"Time for {name}: {time.time() - start}s")
        cv_scores.index = ['Accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        return cv_scores
