import pandas as pd
import numpy as np
import re
import warnings
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import seaborn as sns

class BaseDataHandler():
    '''
    A base class for handling and manipulating pandas DataFrames.
    '''

    def __init__(self, path:str | None = None, df:pd.DataFrame | None = None, encoding:str='utf-8'):
        
        self.file_path = path
        self.encoding = encoding
        success, e = self.try_init_df(df)
        if not success:
            print(e)
    
    @property
    def og_df(self) -> pd.DataFrame:
        return self.__df
    
    @property
    def df(self) -> pd.DataFrame:
        return self.__curr_df
    
    @df.setter
    def df(self, df) -> None:
        self.__curr_df = df
    
    def df_log(self, base:float='e') -> pd.DataFrame:
        '''
        Apply logarithmic transformation to numeric columns of the DataFrame.
        '''
        numeric_cols = self.df.select_dtypes(include='number').columns
        df_log = self.df.copy()

        if base == 'e':
            df_log[numeric_cols] = df_log[numeric_cols].apply(lambda x: np.log(x.clip(lower=1e-9)))
        else:
            df_log[numeric_cols] = df_log[numeric_cols].apply(lambda x: np.log(x.clip(lower=1e-9)) / np.log(base))

        return df_log

    def get_lines(self, amount=5) -> pd.DataFrame:
        '''
        Get the first or last 'amount' of lines from the original DataFrame.
        '''
        return self.og_df.head(amount) if amount > 0 else self.og_df.tail(amount)

    def print_dataframe(self, full=False):
        '''
        Print the original DataFrame, either fully or just the first 5 rows.
        '''
        row = self.df.shape[0] if full else 5
        return self.df.head(row)
    
    @staticmethod
    def _get_pivot(df:pd.DataFrame, values=None, index=None, columns=None, aggfunc:str="mean") -> pd.DataFrame:
        '''
        Create a pivot table from the original DataFrame.
        '''
        return pd.pivot_table(df, values=values, index=index, columns=columns, aggfunc=aggfunc)
    
    def get_pivot(self, values=None, index=None, columns=None, aggfunc:str="mean") -> pd.DataFrame:
        '''
        Create a pivot table from the original DataFrame.
        '''
        return BaseDataHandler._get_pivot(self.df, values=values, index=index, columns=columns, aggfunc=aggfunc)
    
    def try_get_groupby(self, target_col: str | list[str], col:str) -> tuple[bool, Exception | pd.DataFrame]:
        '''
        Group the original DataFrame by 'target_col' and select 'col'.
        '''
        try:
            tmp_df = self.og_df.groupby(by=target_col)[col].count()
        except Exception as e:
            return False, e
        return True, tmp_df

    def try_init_df(self, df) -> tuple[bool, Exception | None]:
        '''
        Initialize the DataFrame and working DataFrame from a given DataFrame or by reading from a CSV file.
        '''
        try:
            if df is not None:
                self.__df = df
                self.__curr_df = df
            else:
                self.__df = pd.read_csv(self.file_path, encoding=self.encoding)
                self.__curr_df = self.og_df.copy()
        except Exception as e:
            return False, e
        return True, None
    
    def try_reset_df(self) -> tuple[bool, Exception | None]:
        '''
        Reset the working DataFrame to the original DataFrame.
        '''
        try:
            self.__curr_df = self.og_df.copy()
        except Exception as e:
            return False, e
        return True, None
    
    def try_update_og_df(self) -> tuple[bool, Exception | None]:
        '''
        Update the original DataFrame to match the current working DataFrame.
        '''
        try:
            self.__df = self.__curr_df.copy()
        except Exception as e:
            return False, e
        return True, None
    
    def try_order_by(self, cols:str | list[str], ascending:bool | list[bool]=True) -> tuple[bool, Exception | None]:
        '''
        Order the original DataFrame by specified columns.
        '''
        try:
            self.__curr_df = self.og_df.sort_values(by=cols, ascending=ascending).reset_index()
        except Exception as e:
            return False, e
        return True, None

    def try_fill_nan(self, use_mean:bool = True) -> tuple[bool, Exception | None]:
        '''
        Fill NaN values in the original DataFrame with either 0 or the mean of each column.
        '''
        try:
            self.__curr_df = self.og_df.fillna(0 if use_mean else self.og_df.mean(numeric_only=True))
        except Exception as e:
            return False, e
        return True, None
    
    @staticmethod
    def _try_add_col(df:pd.DataFrame, target_col:str, func, axis:int=1) -> tuple[bool, Exception | None]:
            '''
            Add a new column to the working DataFrame based on a func function applied to each row or column.
            '''
            try:
                df[target_col] = df.apply(func, axis=axis)
            except Exception as e:
                return False, e
            return True, None
    
    def try_add_col(self, target_col:str, func, axis:int=1) -> tuple[bool, Exception | None]:
        '''
        Add a new column to the working DataFrame based on a func function applied to each row or column.
        '''
        return BaseDataHandler._try_add_col(self.df, target_col, func, axis)
    
    def try_remove_duplicates(self) -> tuple[bool, Exception | None]:
        '''
        Remove duplicate rows from the original DataFrame.
        '''
        try:
            self.__curr_df = self.og_df.drop_duplicates()
        except Exception as e:
            return False, e
        return True, None
    
    def try_save_to_csv(self) -> tuple[bool, Exception | None]:
        '''
        Save the original DataFrame to a new CSV file.
        '''
        try:
            new_file_path = self.file_path.replace('.csv', '_new.csv')
            self.og_df.to_csv(new_file_path)
        except Exception as e:
            return False, e
        return True, None
    
    def try_drop_nan(self, cols:str|list[str]) -> tuple[bool, Exception | None]:
        '''
        Drop rows with NaN values in specified columns and drop columns that are entirely NaN.
        '''
        try:
            self.__curr_df = self.og_df.dropna(subset=cols)
            self.__curr_df = self.og_df.dropna(axis=1, how='all')
        except Exception as e:
            return False, e
        return True, None
    
    def try_clamp_cols(self, cols:str|list[str], lower_bounds:float=0, upper_bounds:float=200, use_og:bool = False) -> tuple[bool, Exception | None]:
        '''
        Clamp the values in specified columns to be within given lower and upper bounds.
        '''
        try:
            self.__curr_df[cols] = self.og_df[cols].clip(lower_bounds,upper_bounds) if use_og else self.__curr_df[cols].clip(lower_bounds,upper_bounds)
        except Exception as e:
            return False, e
        return True, None
    
    def df_norm(self, method:str="minmax") -> pd.DataFrame:
        '''
        Normalize numeric columns of the DataFrame using specified method ('minmax' or 'zscore').
        '''
        numeric_cols = self.df.select_dtypes(include='number').columns
        df_norm = self.df.copy()
        if method == "minmax":
            df_norm[numeric_cols] = df_norm[numeric_cols].apply(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
        elif method == "zscore":
            df_norm[numeric_cols] = df_norm[numeric_cols].apply(
                lambda x: (x - x.mean()) / x.std(ddof=0)
            )
        return df_norm

    def try_clean_string_to_number_col(self, col: str | list[str]) -> tuple[bool, Exception | None]:
        '''
        Clean a column containing string representations of numbers by extracting numeric values
        and converting them to floats. Handles ranges, plus signs, decimals.
        '''
        def helper(s):
            s = str(s).replace(",", "")
            # Match integers or decimals, ignore minus signs as separators
            nums = re.findall(r"\d+(?:\.\d+)?", s)
            if nums:
                nums = list(map(float, nums))
                # If multiple numbers (range), take the mean
                return float(np.mean(nums))
            return None
        try:
            if isinstance(col, str):
                self.__curr_df[col] = self.df[col].apply(helper)
            elif isinstance(col, list):
                for c in col:
                    self.__curr_df[c] = self.df[c].apply(helper)
        except Exception as e:
            return False, e
        return True, None
    
    def try_clean_column_names(self, inplace = True) -> tuple[bool, Exception | pd.DataFrame]:
        '''
        Local version: cleans column names on self.df.
        '''
        state, clean_df = BaseDataHandler._try_clean_column_names(self.df)
        if inplace and state:
            self.df = clean_df
            return True, None
        return state, clean_df

    @staticmethod
    def _try_clean_column_names(df) -> tuple[bool, Exception | pd.DataFrame]:
        '''
        Static version: cleans column names on any DataFrame.
        '''
        def to_snake(name: str) -> str:
            name = name.lower()
            name = re.sub(r'[^a-z0-9]+', '_', name)
            name = name.strip('_')
            return name
        try:
            cleaned_df = df.rename(columns={col: to_snake(col) for col in df.columns})
        except Exception as e:
            return False, e
        return True, cleaned_df

    def try_rename_col(self, col: str | list[str], name: str | list[str], inplace=True) -> tuple[bool, Exception | pd.DataFrame | None]:
        '''
        Local version: renames columns on self.df.
        '''
        s, res = BaseDataHandler._try_rename_col(self.df, col, name)
        if inplace:
            self.df = res.copy()
            return s, None
        return s, res

    @staticmethod
    def _try_rename_col(df, col: str | list[str], name: str | list[str]) -> tuple[bool, Exception | pd.DataFrame]:
        '''
        Static version: renames columns on any DataFrame.
        '''
        try:
            if isinstance(col, str) and isinstance(name, str):
                mapping = {col: name}
            elif isinstance(col, list) and isinstance(name, list):
                mapping = dict(zip(col, name))
            renamed_df = df.rename(columns=mapping)
            return True, renamed_df
        except Exception as e:
            return False, e


    def get_outliers_df(self, method: str = "iqr", lower_percentile: float = 0.01, upper_percentile: float = 0.99) -> pd.DataFrame:
        return BaseDataHandler._get_outliers_df(self.df, method, lower_percentile, upper_percentile)
    
    @staticmethod
    def _get_outliers_df(df, method: str = "iqr", lower_percentile: float = 0.01, upper_percentile: float = 0.99) -> pd.DataFrame:
        """
        Detect outliers for every numeric column in the DataFrame.
        Supports 'iqr', 'zscore', and 'percentile' methods.
        Returns a DataFrame with boolean flags for each column.
        """
        outlier_flags = pd.DataFrame(index=df.index)

        for col in df.select_dtypes(include="number").columns:
            series = df[col]

            if method == "zscore":
                z_scores = (series - series.mean()) / series.std()
                outlier_flags[col] = (z_scores.abs() > 3)

            elif method == "percentile":
                lower = series.quantile(lower_percentile)
                upper = series.quantile(upper_percentile)
                outlier_flags[col] = (series < lower) | (series > upper)

            else:
                if method != "iqr":
                    warnings.warn("Unknown method. Defaulting to IQR.", UserWarning)
                Q1, Q3 = series.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                outlier_flags[col] = (series < lower) | (series > upper)

        return outlier_flags
    
    def try_get_numeric_cols(self) -> pd.DataFrame | None:
        try:
            numeric_cols = self.df.select_dtypes(include="number").columns
        except Exception as e:
            print(e)
            return None
        return self.df[numeric_cols]
    
    def get_outlier_case_study(self, cols=3, width_mul=1,size_mul=1, method="percentile") -> tuple[figure.Figure, np.ndarray]:
        outlier_flags = self.get_outliers_df(
            method=method, lower_percentile=0.01, upper_percentile=0.99
        )

        numeric_cols = self.try_get_numeric_cols()

        n_rows = (len(numeric_cols) + 2) // cols
        fig, axes = plt.subplots(
            n_rows, cols, figsize=(18*size_mul*width_mul, 5*n_rows*size_mul)
        )
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.boxplot(data=self.df, x=col, color="steelblue", ax=axes[i])

            # Overlay outliers in red (only if they exist)
            new_mask = outlier_flags[col]
            if new_mask.any():
                sns.scatterplot(
                    x=self.df[col][new_mask],
                    y=[-0.05] * new_mask.sum(),   # jitter so they appear just above the box
                    color="red",
                    marker="o",
                    ax=axes[i],
                    label="outliers"
                )
                axes[i].legend()

            axes[i].set_title(f"Outliers in {col} ({method})", fontsize=12)

        # Remove unused subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        return fig, axes
    
    def try_clean_string_per_cols(self, cols:str|list[str], inplace:bool=True) -> tuple[bool, Exception | pd.DataFrame]:
        try:
            tmp_df = self.df.copy()
            tmp_df[cols] = (
            self.df[cols]
                .astype(str)              # ensure strings
                .str.strip()              # remove leading/trailing whitespace
                .str.replace(r"\s+", " ", regex=True)  # normalize multiple spaces
                .str.title()              # consistent capitalization
            )
        except Exception as e:
            return False, e
        if inplace :
            self.df[cols] = tmp_df[cols]
            return True, None
        return True, tmp_df
