import pandas as pd
import re

def extract_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne 'tags' au DataFrame contenant les hashtags extraits de la colonne 'description'.

    Args:
        df (pd.DataFrame): DataFrame contenant une colonne 'description'.

    Returns:
        pd.DataFrame: DataFrame avec une colonne 'tags' contenant une liste de hashtags (sans le #).
    """
    # Expression régulière précompilée pour efficacité
    hashtag_pattern = re.compile(r"#(\w+)", flags=re.UNICODE)

    # Appliquer la fonction vectorisée
    df["tags"] = df["description"].astype(str).apply(lambda x: hashtag_pattern.findall(x))

    return df