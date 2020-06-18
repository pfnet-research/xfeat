"""Helper functions."""
from typing import List

from xfeat.types import XDataFrame


def aggregation(
    input_df: XDataFrame,
    group_key: str,
    group_values: List[str],
    agg_methods: List[str],
):
    """Aggregate values after grouping table rows by a given key.

    Arguments:
        input_df (XDataFrame) : Input data frame.
        group_key (str) : Used to determine the groups for the groupby.
        group_values (List[str]) : Used to aggregate values for the groupby.
        agg_methods (List[str]) : List of function names, e.g. ['mean', 'max', 'min', 'std'].
    Returns:
        Tuple[XDataFrame, List[str]] : Tuple of output dataframe and new column names.
    """
    new_df = input_df.copy()

    new_cols = []
    for agg_method in agg_methods:
        for col in group_values:
            new_col = f"agg_{agg_method}_{col}_grpby_{group_key}"

            # NOTE(smly):
            # Failed when cudf.DataFrame try to merge with cudf.Series.
            # Use workaround to merge with cudf.DataFrame.
            # Ref: http://github.com/rapidsai/cudf/issues/5013
            df_agg = (
                input_df[[col] + [group_key]].groupby(group_key)[[col]].agg(agg_method)
            )
            df_agg.columns = [new_col]
            new_cols.append(new_col)
            new_df = new_df.merge(
                df_agg, how="left", right_index=True, left_on=group_key
            )

    return new_df, new_cols
