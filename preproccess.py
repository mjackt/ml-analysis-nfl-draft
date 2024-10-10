import pandas as pd

def process_file():
    picks = pd.read_csv('draft_picks.csv')

    df = pd.DataFrame().assign(Player=picks['pfr_player_name'],
                            Round=picks['round'],
                            Pick=picks['pick'],
                            Position=picks['position'],
                            Category=picks['category'],
                            WAV=picks['w_av'],
                            Games=picks['games'])
    ###   Duplicate Checks   ###
    df.drop_duplicates(inplace=True)
    
    ###   Null checks and transformation   ###
    print(df.shape)
    print(df.isnull().sum())

    print("\nNumber of records with null games AND null wAV")
    print(sum((df['WAV'].isna()) & (df['Games'].isna())))

    #Transformation
    df['WAV'].fillna(0, inplace=True)
    df['Games'].fillna(0, inplace=True)

    ###   Error checks   ###
    assert((df['WAV'] % 1  == 0).all())
    assert((df['Round'] % 1  == 0).all())
    assert((df['Pick'] % 1  == 0).all())
    assert((df['Games'] % 1  == 0).all())

    ###   Outlier Checks   ###
    print(df.describe())

    #TALKING POINTS:
    #Sides had several nulls so i removed it because it wasn't essential
    #WAV was essentiall and had several thouscounand null values. However it just turned out to be those who never played. Essentially MNAR data

    #Do i chop off later pick due to draft being smaller now than in 1980
    
    #Manually updated sides column due to small number of nulls
    
    return df

if __name__ == '__main__':
    df = process_file()

    print("\n")
    print(df.head())
    print(df.shape)
    print(df.info())

    print(df['Category'].explode().unique())