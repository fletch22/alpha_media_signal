from ams.config import constants

from ams.utils import pandas_utils


def test_config():
  # Arrange
  # Act
  creds = constants.FLETCH22_CREDS

  print(creds.api_key)

  # Assert
  assert (len(creds.api_key) > 10)


COLS = ['location_Alaska; U.S.A', 'location_Czech Republic', 'industry_Confectioners',
        'location_Oklahoma; U.S.A', 'location_Indonesia', 'table_SF3B', 'pe1',
        'industry_Apparel Retail', 'close_SMA_100_days_since_under',
        'famaindustry_Fabricated Products', 'industry_Data Storage', 'location_Oregon; U.S.A',
        'location_Germany', 'location_Delaware; U.S.A', 'industry_Lodging',
        'location_Israel-Jordan', 'currency_IDR', 'industry_Security & Protection Services',
        'famaindustry_Candy & Soda', 'location_Philippines',
        'famaindustry_Electrical Equipment', 'famaindustry_Chemicals', 'de',
        'location_West Virginia; U.S.A', 'location_British Virgin Islands',
        'industry_Shipping & Ports', 'industry_Beverages - Non-Alcoholic', 'roe',
        'industry_Utilities - Regulated Gas', 'industry_Grocery Stores',
        'location_Netherlands Antilles', 'industry_Thermal Coal',
        'industry_REIT - Hotel & Motel', 'f22_has_cashtag', 'location_United Kingdom',
        'currency_ARS', 'location_Jersey', 'location_Isle Of Man',
        'close_SMA_15_days_since_under', 'debtc', 'famaindustry_Real Estate',
        'industry_Banks - Regional', 'close_SMA_20', 'location_Canada (Federal Level)',
        'close_SMA_100', 'famaindustry_Coal', 'user_geo_enabled', 'industry_Entertainment',
        'industry_Staffing & Outsourcing Services', 'industry_Advertising Agencies',
        'user_protected', 'sector_Energy', 'famaindustry_Precious Metals',
        'famaindustry_Steel Works Etc', 'location_Washington; U.S.A', 'location_<unknown>',
        'accoci', 'industry_Health Information Services', 'location_North Carolina; U.S.A',
        'location_Malta', 'category_<unknown>', 'liabilitiesc', 'famaindustry_Healthcare',
        'industry_Silver', 'taxexp', 'category_ETF', 'industry_Engineering & Construction',
        'location_Saskatchewan; Canada', 'scalerevenue_1 - Nano', 'location_Nevada; U.S.A',
        'industry_Semiconductor Memory', 'location_Arizona; U.S.A',
        'famaindustry_Business Supplies', 'lastupdated_eq_fun', 'currency_DKK',
        'industry_Copper', 'industry_Uranium', 'close_SMA_15', 'industry_Aerospace & Defense',
        'industry_Electronic Components', 'f22_sentiment_pos', 'ev', 'scalerevenue_3 - Small',
        'location_China', 'location_Colombia', 'location_Tennessee; U.S.A', 'open',
        'f22_ticker', 'industry_Trucking', 'user_listed_count', 'payoutratio',
        'location_Austria', 'industry_Insurance - Specialty', 'ncfdebt',
        'industry_Household & Personal Products', 'f22_compound_score', 'f22_ticker_in_text',
        'famaindustry_Transportation', 'purchase_date', 'industry_Recreational Vehicles',
        'deferredrev', 'pe', 'famaindustry_Wholesale', 'currency_JPY', 'location_Taiwan',
        'industry_Financial Data & Stock Exchanges', 'ros', 'location_New Hampshire; U.S.A',
        'famaindustry_Insurance', 'industry_Waste Management',
        'industry_Farm & Heavy Construction Machinery', 'industry_Tobacco', 'retearn',
        'category_Domestic Preferred Stock', 'location_Wisconsin; U.S.A',
        'location_Arkansas; U.S.A', 'location_France', 'famaindustry_Construction Materials',
        'revenueusd', 'industry_Oil & Gas E&P', 'industry_Packaged Foods',
        'industry_Banks - Global', 'location_Mauritius', 'famaindustry_Apparel',
        'location_Connecticut; U.S.A', 'invcap', 'industry_Home Improvement Retail',
        'famaindustry_Agriculture', 'f22_is_tweet_after_hours', 'taxliabilities', 'tbvps',
        'famaindustry_Medical Equipment', 'industry_Insurance - Diversified', 'currency_USD',
        'sicsector_Transportation Communications Electric Gas And Sanitary Service',
        'industry_Beverages - Soft Drinks', 'user_followers_count', 'epsusd',
        'location_Kentucky; U.S.A', 'industry_Insurance - Life',
        'location_South Carolina; U.S.A', 'sector_Financial Services', 'currency_NOK',
        'location_Massachusetts; U.S.A', 'close_SMA_20_days_since_under',
        'location_Saint Vincent And The Grenadines', 'deposits',
        'industry_Oil & Gas Equipment & Services', 'industry_Integrated Freight & Logistics',
        'volume', 'industry_Personal Services', 'industry_Specialty Industrial Machinery',
        'currency_ILS', 'famaindustry_Utilities', 'industry_Medical Devices',
        'location_New Jersey; U.S.A', 'location_Greece', 'location_Ontario; Canada',
        'investmentsc', 'location_Unknown', 'category_Canadian Common Stock Primary Class',
        'famaindustry_Pharmaceutical Products', 'industry_Metal Fabrication',
        'location_Netherlands', 'industry_Utilities - Regulated Water',
        'industry_Farm & Construction Equipment', 'f22_sentiment_neu', 'sicsector_Retail Trade',
        'sharefactor', 'location_Minnesota; U.S.A', 'industry_REIT - Residential', 'debtnc',
        'industry_REIT - Specialty', 'ebt', 'industry_REIT - Healthcare Facilities',
        'location_Hong Kong', 'dividends', 'location_Georgia U.S.A.', 'close_SMA_50', 'ncfx',
        'divyield', 'currency_CHF', 'industry_Utilities - Renewable', 'currency_AUD',
        'assetturnover', 'industry_Other Industrial Metals & Mining', 'opinc',
        'sector_Communication Services', 'currency_KRW', 'industry_Specialty Finance',
        'location_Bermuda', 'currency_CLP', 'industry_Lumber & Wood Production', 'ebitusd',
        'famaindustry_Personal Services', 'ppnenet', 'industry_Education & Training Services',
        'capex', 'famaindustry_Shipping Containers', 'location_Maryland; U.S.A',
        'industry_Computer Hardware', 'industry_Credit Services', 'location_Ireland',
        'sicsector_Construction', 'famaindustry_Textiles', 'industry_Farm Products',
        'industry_Steel', 'location_Nebraska; U.S.A', 'f22_id', 'location_Cyprus',
        'scalemarketcap_6 - Mega', 'location_Colorado; U.S.A', 'assetsc',
        'industry_Insurance - Reinsurance', 'industry_Health Care Plans',
        'location_Manitoba; Canada', 'scalemarketcap_4 - Mid', 'user_verified',
        'scalerevenue_5 - Large', 'ebitdausd', 'famaindustry_Defense',
        'industry_Internet Retail', 'location_Brazil', 'f22_num_other_tickers_in_tweet',
        'industry_Auto Parts', 'location_Iceland', 'industry_Business Services',
        'industry_Beverages - Wineries & Distilleries', 'currency_CNY', 'location_Utah; U.S.A',
        'location_Sweden', 'famaindustry_Printing and Publishing', 'industry_Apparel Stores',
        'scalemarketcap_2 - Micro', 'liabilities', 'sicsector_<unknown>',
        'location_Alberta; Canada', 'location_Maine; U.S.A', 'famaindustry_Consumer Goods',
        'taxassets', 'revenue', 'calendardate', 'ebit', 'famaindustry_Trading',
        'industry_Furnishings Fixtures & Appliances', 'industry_Financial Conglomerates',
        'location_Israel-Syria', 'industry_Apparel Manufacturing', 'location_Texas; U.S.A',
        'industry_Footwear & Accessories', 'currency_SEK', 'location_Hawaii; U.S.A',
        'industry_Oil & Gas Drilling', 'assets', 'location_Finland', 'industry_Aluminum',
        'location_Louisiana; U.S.A', 'industry_Beverages - Brewers',
        'industry_Banks - Regional - US', 'industry_Broadcasting - Radio',
        'location_Republic Of Korea', 'sicsector_Manufacturing', 'sicsector_Nonclassifiable',
        'famaindustry_Machinery', 'location_Newfoundland; Canada', 'location_Poland',
        'industry_Medical Instruments & Supplies', 'industry_Coking Coal', 'ncf',
        'user_follow_request_sent', 'industry_Pharmaceutical Retailers', 'famaindustry_Retail',
        'category_ETN', 'industry_Drug Manufacturers - Specialty & Generic',
        'category_ADR Preferred Stock', 'investments', 'sbcomp', 'cashneq', 'currency_TWD',
        'possibly_sensitive', 'famaindustry_Beer & Liquor',
        'industry_Information Technology Services', 'currency_INR', 'sector_Technology',
        'location_Costa Rica', 'location_District Of Columbia; U.S.A', 'location_South Africa',
        'location_Virginia; U.S.A', 'location_Luxembourg', 'famaindustry_Business Services',
        'marketcap', 'location_British Columbia; Canada', 'future_high', 'sicsector_Mining',
        'days_util_sale', 'category_Domestic Common Stock', 'sector_Consumer Defensive',
        'dimension', 'rnd', 'location_Malaysia', 'original_close_price',
        'industry_REIT - Diversified', 'bvps', 'famaindustry_Electronic Equipment',
        'industry_Utilities - Regulated Electric', 'epsdil', 'location_Belgium', 'netinccmn',
        'depamor', 'industry_Real Estate - Diversified', 'industry_REIT - Industrial',
        'famaindustry_Restaraunts Hotels Motels', 'cor', 'retweet_count',
        "location_Democratic People'S Republic Of Korea", 'industry_Auto & Truck Dealerships',
        'netinccmnusd', 'industry_Drug Manufacturers - Major', 'location_Michigan; U.S.A',
        'evebit', 'category_Canadian Preferred Stock', 'sicsector_Services',
        'industry_Home Improvement Stores', 'famasector',
        'industry_Electronic Gaming & Multimedia', 'category_ETD',
        'industry_Industrial Metals & Minerals', 'currency_GBP', 'industry_Marine Shipping',
        'location_Denmark', 'location_Peru', 'industry_Diversified Industrials',
        'industry_Long-Term Care Facilities', 'currency_ZAR', 'location_Montana; U.S.A',
        'location_Canada', 'location_South Dakota; U.S.A', 'category_Canadian Common Stock',
        'industry_Biotechnology', 'currency_TRY', 'f22_sentiment_compound',
        'famaindustry_Non-Metallic and Industrial Metal Mining', 'user_statuses_count',
        'inventory', 'industry_Travel Services', 'industry_Insurance - Property & Casualty',
        'famaindustry_Entertainment', 'currency_PLN', 'industry_Business Equipment & Supplies',
        'currency_BRL', 'assetsnc', 'industry_Staffing & Employment Services',
        'industry_Electrical Equipment & Parts', 'ncfi', 'industry_Diagnostics & Research',
        'netincdis', 'industry_Packaging & Containers', 'currency_CAD', 'location_Bahamas',
        'f22_sentiment_neg', 'location_Ghana', 'assetsavg', 'location_Italy',
        'location_Guernsey', 'ncfcommon', 'ebitda', 'category_ADR Stock Warrant',
        'famaindustry_Automobiles and Trucks', 'currency_MYR',
        'category_Domestic Common Stock Primary Class', 'location_Saudi Arabia',
        'industry_Real Estate Services', 'scalerevenue_4 - Mid',
        'industry_Electronics & Computer Distribution', 'netmargin', 'invcapavg', 'evebitda',
        'low', 'location_Hungary', 'industry_Paper & Paper Products', 'location_Oman', 'dps',
        'industry_Infrastructure Operations', 'reportperiod',
        'industry_Oil & Gas Refining & Marketing',
        'industry_Utilities - Independent Power Producers', 'sector_Healthcare',
        'location_Indiana; U.S.A', 'location_Switzerland',
        'industry_Other Precious Metals & Mining', 'famaindustry_Computers',
        'industry_Drug Manufacturers - General', 'location_Australia', 'consolinc',
        'sector_Consumer Cyclical', 'future_low', 'ncfbus', 'scalemarketcap_3 - Small',
        'industry_Gold', 'currency_COP', 'industry_Tools & Accessories', 'industry_Publishing',
        'industry_Scientific & Technical Instruments', 'location_Vermont; U.S.A',
        'location_Turkey', 'currency_HKD', 'famaindustry_Banking',
        'industry_Residential Construction', 'industry_Consulting Services',
        'industry_Semiconductors', 'industry_Airports & Air Services', 'location_Japan',
        'industry_Agricultural Inputs', 'industry_Semiconductor Equipment & Materials',
        'industry_Communication Equipment', 'eps', 'table_SF1', 'location_Iowa; U.S.A', 'ps',
        'sgna', 'fcfps', 'scalemarketcap_1 - Nano', 'industry_Department Stores', 'netinc',
        'close', 'location_New Mexico; U.S.A', 'sicsector_Agriculture Forestry And Fishing',
        'category_ADR Common Stock Secondary Class', 'industry_Resorts & Casinos',
        'category_ADR Common Stock', 'industry_Airlines', 'sps', 'liabilitiesnc',
        'location_Mexico', 'ebitdamargin', 'location_Argentina', 'grossmargin',
        'industry_Healthcare Plans', 'location_Russian Federation', 'currency_PEN',
        'famaindustry_Tobacco Products', 'scalerevenue_<unknown>', 'roic', 'roa',
        'industry_Coal', 'location_Idaho; U.S.A', 'location_Alabama; U.S.A',
        'industry_Financial Exchanges', 'intexp', 'industry_Medical Care', 'currency_MXN',
        'industry_Solar', 'fcf', 'category_Domestic Stock Warrant', 'location_Ohio; U.S.A',
        'location_Wyoming; U.S.A', 'famaindustry_Construction', 'sector_Basic Materials',
        'famaindustry_Communication', 'industry_Food Distribution', 'location_Venezuela',
        'location_India', 'currency_EUR', 'industry_Computer Systems', 'location_Guam',
        'tangibles', 'location_Gibraltar', 'industry_Industrial Distribution',
        'industry_Shell Companies', 'location_Virgin Islands; U.S.', 'location_New Zealand',
        'location_Cayman Islands', 'location_Uruguay',
        'sicsector_Finance Insurance And Real Estate', 'industry_Conglomerates',
        'industry_Rental & Leasing Services', 'location_Rhode Island; U.S.A',
        'famaindustry_Recreation', 'payables', 'famaindustry_Shipbuilding Railroad Equipment',
        'ps1', 'netincnci', 'price', 'location_Macau',
        'famaindustry_Rubber and Plastic Products', 'intangibles', 'favorite_count',
        'future_open', 'location_Nova Scotia; Canada', 'cashnequsd',
        'location_Mississippi; U.S.A', 'industry_REIT - Retail', 'industry_Specialty Chemicals',
        'industry_Capital Markets', 'sicsector_Wholesale Trade', 'industry_Discount Stores',
        'gp', 'industry_Business Equipment', 'industry_Home Furnishings & Fixtures',
        'industry_Gambling', 'opex', 'table_SEP', 'industry_Leisure',
        'location_United Republic Of Tanzania', 'ncfo', 'debtusd', 'industry_REIT - Mortgage',
        'location_Thailand', 'location_Panama', 'user_friends_count', 'prefdivis', 'closeunadj',
        'industry_Textile Manufacturing', 'industry_Broadcasting', 'debt',
        'category_ADR Common Stock Primary Class', 'location_California; U.S.A',
        'close_SMA_50_days_since_under', 'location_Illinois; U.S.A',
        'user_has_extended_profile', 'industry_Real Estate - General',
        'industry_Medical Care Facilities', 'receivables', 'sector_Real Estate', 'category_IDX',
        'equityavg', 'currentratio', 'fxusd', 'scalemarketcap_<unknown>', 'investmentsnc',
        'table_SFP', 'industry_Pollution & Treatment Controls',
        'location_New Brunswick; Canada', 'location_Maldives',
        'famaindustry_Petroleum and Natural Gas', 'shareswadil', 'location_North Dakota; U.S.A',
        'currency_VEF', 'future_date', 'industry_Oil & Gas Midstream',
        'location_Florida; U.S.A', 'industry_Telecom Services', 'location_Georgia; U.S.A',
        'currency_NZD', 'industry_Internet Content & Information', 'ncff', 'datekey',
        'category_Domestic Common Stock Secondary Class', 'location_Singapore',
        'workingcapital', 'industry_Software - Infrastructure', 'famaindustry_Food Products',
        'location_Israel', 'sector_Utilities', 'future_close', 'user_is_translation_enabled',
        'scalerevenue_6 - Mega', 'scalerevenue_2 - Micro', 'location_United Arab Emirates',
        'buy_sell', 'equityusd', 'sharesbas', 'siccode', 'equity', 'ncfdiv',
        'sicsector_Public Administration', 'industry_Railroads', 'sector_<unknown>',
        'location_Kansas; U.S.A', 'industry_Media - Diversified', 'stock_val_change',
        'location_Pennsylvania; U.S.A', 'table_<unknown>', 'industry_Real Estate - Development',
        'user_screen_name', 'famaindustry_Measuring and Control Equipment',
        'industry_Specialty Business Services', 'pb', 'famaindustry_Almost Nothing',
        'industry_Utilities - Diversified', 'industry_Banks - Diversified', 'currency_RUB',
        'sector_Industrials', 'industry_Medical Distribution', 'industry_Mortgage Finance',
        'industry_Luxury Goods', 'industry_Savings & Cooperative Banks',
        'industry_Asset Management', 'industry_Auto Manufacturers', 'currency_PHP',
        'location_Marshall Islands', 'location_Chile', 'scalemarketcap_5 - Large', 'ncfinv',
        'close_SMA_200', 'high', 'industry_Restaurants', 'location_Monaco',
        'currency_<unknown>', 'industry_Broadcasting - TV', 'location_Jordan',
        'location_New York; U.S.A', 'industry_Furnishings', 'location_Missouri; U.S.A',
        'industry_<unknown>', 'industry_Oil & Gas Integrated', 'location_Quebec; Canada',
        'industry_Specialty Retail', 'location_United States; U.S.A', 'location_Spain',
        'category_Canadian Stock Warrant', 'industry_Chemicals', 'famaindustry_Aircraft',
        'close_SMA_200_days_since_under', 'industry_REIT - Office',
        'industry_Software - Application', 'industry_Consumer Electronics',
        'industry_Building Products & Equipment', 'location_Puerto Rico', 'location_Norway',
        'category_CEF', 'famaindustry_<unknown>', 'industry_Insurance Brokers',
        'industry_Building Materials', 'shareswa', 'f22_day_tweet_count']


def test_find_columns_with_value():
  import pandas as pd
  df = pd.DataFrame([{"foo": "abc", "bar": "def"}])

  # Act
  cols = pandas_utils.find_columns_with_value(df=df, value_to_find='abc')

  # Assert
  assert (len(cols) == 1)
  assert (cols[0] == "foo")


def test_foo():
  cols = [c for c in COLS if not c.startswith("location_")
          and not c.startswith("close_")
          and not c.startswith("industry_")
          and not c.startswith("famaindustry_")
          and not c.startswith("currency_")
          and not c.startswith("user_")
          and not c.startswith("category_")
          and not c.startswith("sector_")
          and not c.startswith("sicsector_")
          and not c.startswith("scalemarketcap_")
          and not c.startswith("scalerevenue_")
          ]

  print(len(cols))
  print(cols)

  import pandas as pd
  df = pd.DataFrame([{"foo": "abc", "bar": "def"}])

  print(f"Columns matched: {pandas_utils.find_columns_with_value(df=df, value_to_find='abc')}")
