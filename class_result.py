def append_to_classification_report(column_name, y_test, y_pred,filename = 'data/pa_classification_report.csv'):
    import pandas as pd
    import os
    from sklearn.metrics import classification_report
    
    
    df_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    df_report.rename(columns={'f1-score': column_name}, inplace=True)
    
    if os.path.isfile(filename):
        df = pd.read_csv(filename, index_col=0)
        df = df.drop(columns=[column_name], errors='ignore')
        df = df.join(df_report[[column_name]])
    else:
        df = df_report[[column_name]]

    df.to_csv(filename, index=True)
    return df