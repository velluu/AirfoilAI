import pandas as pd
import numpy as np

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.copy()
    feature_df['log_Re'] = np.log10(feature_df['Re'])
    
    feature_cols = ['camber', 'camber_pos', 'thickness', 'Mach', 'log_Re', 'alpha']
    X = feature_df[feature_cols]
    y = feature_df['L_D']
    
    return pd.concat([X, y], axis=1)
