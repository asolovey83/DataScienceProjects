import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Variables
RANDOM_SEED = 42
VAL_SIZE = 0.15

# A target metric function
def mape(y_true, y_pred):    
    return np.mean(np.abs((y_pred-y_true)/y_true))

# A function for outliers preprocessing
def outliers_replacement(df, column, method='median'):
    '''
    Replaces outliers in the series with the specific value 
    method='median' - replace with median
    method='average' - replace with mean
    method ='probable' - random distribution
    '''
    IQR = df[column].quantile(0.75) - df[column].quantile(0.25)
    perc25 = df[column].quantile(0.25)
    perc75 = df[column].quantile(0.75)

    f = perc25 - 1.5*IQR
    l = perc75 + 1.5*IQR

    if method =='median':
        df.loc[(df[column] < f) | (df[column] > l), column] = df[column].median()
    elif method =='average':
        df.loc[(df[column] < f) | (df[column] > l), column] = df[column].mean()
    elif method =='probable':
        # replacing outliers to nan 
        df[column] = np.where((df[column] < f) | (df[column] > l), np.nan, df[column])
        mask = df[column].isna()
        # distribution stats
        p = df[column].value_counts() / len(df[column].dropna())
        # filling missing values with the probability `p`
        df.loc[mask, column] = np.random.choice(p.index.to_list(),
                                            size=mask.sum(), 
                                            p=p.to_list())


df = pd.read_csv('land_lots_eda_available.csv')
y = df['pricePerOne']

df.drop(['id'], axis = 1, inplace = True)
df.drop(['price'], axis = 1, inplace = True)
df.drop(['isAvailable'], axis = 1, inplace = True)

df['label'] = 1

st.title('Land Lot Price Prediction')
st.markdown("##### This application allows you to predict a relevant land lot's price per 1 ha based on the lot's features. You can then compare the real price of the lot set by the owner with the predicted price and make decision about purchase")
st.info("##### Please enter all the features of your lot, wait until the app does its work and then you`ll see the result in the bottom")
st.warning("##### This app implies that you have all the real values of the following lot's features at hand as they're named below. If you're just testing the app and not taking into account dependencies between the features, the model may provide incorrect prediction")

# Status
status = st.radio("What is your lot's status", (3,4))


# Koatuu
koatuu = st.selectbox("Your lot's Koatuu", ['Іванків, Іванківська, Вишгородський, Київська, Україна',
       'Іванківська, Вишгородський, Київська, Україна',
       'Подо-Калинівка, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Зеленопідська, Каховський, Херсонська, Україна',
       'Урожайне, Бериславська, Бериславський, Херсонська, Україна',
       'Раківка, Бериславська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Демидівка, Демидівська, Дубенський, Рівненська, Україна',
       'Рудка, Демидівська, Дубенський, Рівненська, Україна',
       'Мар’їнська, Покровський, Донецька, Україна',
       'Придніпровське, Червоногригорівська, Нікопольський, Дніпропетровська, Україна',
       'Верхня Ланна, Ланнівська, Полтавський, Полтавська, Україна',
       'Нововасилівка, Снігурівська, Баштанський, Миколаївська, Україна',
       'Середино-Будська, Шосткинський, Сумська, Україна',
       'Нововасилівка, Зноб-Новгородська, Шосткинський, Сумська, Україна',
       'Боромлянська, Охтирський, Сумська, Україна',
       'Томарине, Бериславська, Бериславський, Херсонська, Україна',
       'Інгульська, Баштанський, Миколаївська, Україна',
       'Калинівка, Березанська, Миколаївський, Миколаївська, Україна',
       'Новопсковська, Старобільський, Луганська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Лохвицька, Миргородський, Полтавська, Україна',
       'Чорнухинська, Лубенський, Полтавська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Андріяшівська, Роменський, Сумська, Україна',
       'Градизька, Кременчуцький, Полтавська, Україна',
       'Адампіль, Старосинявська, Хмельницький, Хмельницька, Україна',
       'Тавричанська, Каховський, Херсонська, Україна',
       'Бериславська, Бериславський, Херсонська, Україна',
       'Чернелиця, Чернелицька, Коломийський, Івано-Франківська, Україна',
       'Великоолександрівська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Бродівська, Золочівський, Львівська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Паплинці, Старосинявська, Хмельницький, Хмельницька, Україна',
       'Острівка, Куцурубська, Миколаївський, Миколаївська, Україна',
       'Батуринська, Ніжинський, Чернігівська, Україна',
       'Путивльська, Конотопський, Сумська, Україна',
       'Степанівська, Сумський, Сумська, Україна',
       'Дрімайлівка, Куликівська, Чернігівський, Чернігівська, Україна',
       'Оболонська, Кременчуцький, Полтавська, Україна',
       'Глобинська, Кременчуцький, Полтавська, Україна',
       'Чорноморська, Миколаївський, Миколаївська, Україна',
       'Гребениківка, Боромлянська, Охтирський, Сумська, Україна',
       'Березанська, Миколаївський, Миколаївська, Україна',
       'Ланнівська, Полтавський, Полтавська, Україна',
       'Садівська, Сумський, Сумська, Україна',
       'Кролевецька, Конотопський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Старовірівська, Красноградський, Харківська, Україна',
       'Підлісне, Кіптівська, Чернігівський, Чернігівська, Україна',
       'Горбове, Куликівська, Чернігівський, Чернігівська, Україна',
       'Матіївка, Батуринська, Ніжинський, Чернігівська, Україна',
       'Бугрувате, Чернеччинська, Охтирський, Сумська, Україна',
       'Нечаянська, Миколаївський, Миколаївська, Україна',
       'Ювілейна, Херсонський, Херсонська, Україна',
       'Нижньосірогозька, Генічеський, Херсонська, Україна',
       'Мирненська, Скадовський, Херсонська, Україна',
       'Лубенська, Лубенський, Полтавська, Україна',
       'Лютенська, Миргородський, Полтавська, Україна',
       'Борозенська, Бериславський, Херсонська, Україна',
       'Шевченківська, Куп’янський, Харківська, Україна',
       'Хухра, Чернеччинська, Охтирський, Сумська, Україна',
       'Хрінники, Демидівська, Дубенський, Рівненська, Україна',
       'Августинівка, Широківська, Запорізький, Запорізька, Україна',
       'Сергіївська, Миргородський, Полтавська, Україна',
       'Кардашівка, Чернеччинська, Охтирський, Сумська, Україна',
       'Шосткинська, Шосткинський, Сумська, Україна',
       'Хибалівка, Куликівська, Чернігівський, Чернігівська, Україна',
       'Бакирівка, Чернеччинська, Охтирський, Сумська, Україна',
       'Жигайлівка, Боромлянська, Охтирський, Сумська, Україна',
       'Сенчанська, Миргородський, Полтавська, Україна',
       'Краснолуцька, Миргородський, Полтавська, Україна',
       'Гадяцька, Миргородський, Полтавська, Україна',
       'Качкарівка, Милівська, Бериславський, Херсонська, Україна',
       'Новогродівська, Покровський, Донецька, Україна',
       'Хлібодарівська, Волноваський, Донецька, Україна',
       'Можари, Словечанська, Коростенський, Житомирська, Україна',
       'Скадовка, Чаплинська, Каховський, Херсонська, Україна',
       'Листвин, Словечанська, Коростенський, Житомирська, Україна',
       'Карлівка, Карлівська, Полтавський, Полтавська, Україна',
       'Стягайлівка, Зноб-Новгородська, Шосткинський, Сумська, Україна',
       'Пологи, Чернеччинська, Охтирський, Сумська, Україна',
       'Чернеччина, Краснопільська, Сумський, Сумська, Україна',
       'Рогатин, Рогатинська, Івано-Франківський, Івано-Франківська, Україна'])

print(f'Your status is', koatuu)

# pricePerOne numberinput
lot_pricePerOne = st.number_input('Please, input the real price for ha', 10000, 50000)

# estimatePrice numberinput
estimatePrice = st.number_input('Please, input the estimated price for the whole lot', 10000, 9415117)

# rentRate slider
rentRate = st.slider('Please, select rental rate for the lot', 46, 41165)

# rentRate slider
rentalYield = st.slider('Please, select renatal yield for the lot', 0.2, 7.7)

# purpose radio
purpose = st.radio('What is your lot`s purpose', (0,1))

# ownerEdrpou select
ownerEdrpou = st.selectbox('Choose your lot`s ownerEdrpou', [3578305088, 2284411333, 2520006184, 2762921072, 1518314176,
       3408701198, 2797803591, 1963310215, 1856508382, 3274600270,
       2455910569, 1616305266, 1433704084, 1850110767, 1951910110,
       3031803540, 3155924801, 2472515715, 2283523104, 1091716327,
       2261411476, 2172714889, 2717313833, 2206622700, 1939718124,
       2434710191, 1992116517, 2437008351, 1706925140, 2444707963,
       1636914806, 2557415292, 2047312603, 1894112114, 2022012060,
       2038312921, 2898801012, 2770704258, 2709509144, 1142909362,
       2331819913, 3239015251, 3276308852, 2255605066, 1957106628,
       1927705001, 1201219001, 2755010354, 2120417688, 3344808497,
       2180911609, 2195414318, 2860115552, 2397108623, 2580809499,
       1987021870, 2591204340, 2201923375, 2155809832, 2394908739,
       2520508240, 2298417261, 2205618325, 2928504120, 2014211531,
       2360614905, 2305717493, 2389312850, 2858301844, 3033912891,
       3355000542, 2190410111, 2054002269, 2734306382, 1422117829,
       2812012895, 3245411978, 2966117326, 2699503926, 2567522483,
       3321514194, 1443620643, 2198232628, 2638708163, 2251817743,
       2176219872, 3296307376, 3479709609, 2306422227, 2528811728,
       1882214622, 2332811281, 3532706087, 1910817410, 2448620643,
       2059009936, 1432707586, 2741809958, 2937123271, 2627104567,
       2373303296, 2940916745, 2838103555, 3107214863, 1375202079,
       2118810104, 3743905967, 1703914480, 2854120624, 2769114441,
       2629505994, 3549010092, 2038316559, 2654006151, 2285703733,
                0, 3138710692, 1941802895, 2429915832, 2539200328,
       3104403088, 3027203880, 2448017042, 1529904441, 2301181796,
       2218807090, 2402311208, 2157321026, 1792903344, 1695311297,
       1888211729, 3010625446, 2662519661, 3179200119, 2421308814,
       2403412519, 3313512040, 2089618681, 2166413827, 2010817453,
       2160225508, 2176723982, 2068311700, 1360028227, 2604217987,
       3313304975, 3468510652, 2248405110, 3184017860, 2883812447,
       2795506784, 2533017085, 2480512209, 2165225596, 1889111155,
       2525511938, 3126509483, 1607250945, 2740405848, 2612904382,
       3286411515, 2501304662, 3287909384, 1819702127, 2576119205,
       2772818524, 2800119708, 2632621541, 2613512599, 1444806812,
       3133610137, 2887803701, 2453001838, 2389307252, 2166703283,
       1727113909, 2655512976, 2012117167, 1667904925, 2229316987,
       2411909035, 2996213175, 1308209220, 2163813569, 2862817921,
       2542605953, 2045437472, 3349009358, 2466706490, 2901719503,
       2413404553, 2793310828, 1875005604, 1748705401, 1414624427,
       2370509229, 2783620237, 2505710342, 1694112660, 2550302815,
       1935103129, 2018302975, 2936924915, 2687720689, 2215117530,
       2817222063, 3073112372, 2418207776, 2556825089, 2131005819,
       3523909113, 1891307649, 2370316611, 2797620670, 2681508339,
       1791621629, 1993811988, 2355813375, 2771004191, 2969714900,
       2038708533, 2425412067, 2623115909, 2802220952, 2602504943,
       2453318740, 3200119236, 2883709552, 2861710791, 1723716180,
       2722508444, 2871009554, 3559913334, 2483722709, 2717208823,
       2296510148, 3060017396, 2681509742, 1985310354, 2817509067,
       2046209651, 2440005804, 2486315641, 3048819619, 2340914586,
       1697005323, 2851813241, 1728213074, 2367911449, 1515411420,
       2754408142, 2074419265, 2321419595, 2699704886, 2792021314,
       2570616745, 2176702719, 2194311524, 3239216160, 3182115588,
       2258417723, 2741803480, 3106522390])

# renterCompany select
renterCompany = st.selectbox('Choose your lot`s renterCompany', [15.0, 17.0, 14.0, 19.0, 21.0, 23.0, 22.0, 18.0, 20.0, 16.0, 13.0])

# renterEdrpou select
renterEdrpou = st.selectbox('Choose your lot`s renterEdrpou', [41102844, 41101589, 41099127, 41102163, 41103827, 41105190,
       34264631, 41107067, 41104731, 41104967, 41481188])

# region_id select
region_id = st.selectbox('Select your lot`s region_id', [3222055100, 3222086800, 6525082500, 6525085600, 6523584000,
       6520687300, 6520686600, 6520680200, 5621455300, 5621485000,
       1423385000, 1222986000, 5321681000, 6520683000, 4825782700,
       5924487600, 5924483800, 5924480400, 5925080800, 6520686900,
       6520687100, 4820681200, 4820981800, 4423382500, 5320483000,
       5322686200, 5325183200, 5924185400, 5924189600, 5320686800,
       5924186100, 6824480500, 6523580700, 6520684400, 2621655700,
       5325182600, 6520985000, 6520685200, 4620387000, 5925080400,
       6824485000, 4825182600, 4620381300, 7420385000, 5923883800,
       5924785600, 7422783500, 5324585100, 5320682800, 5325185500,
       4825181200, 5925081600, 4820981200, 5321684900, 5320481400,
       5924787900, 5924786300, 5922681700, 5922980400, 6324283500,
       5922981200, 7422088400, 7422782500, 7420386000, 5920380800,
       5922985800, 4820980900, 4820982400, 4820985000, 6525086000,
       6523884500, 6523287700, 5322881100, 5320485500, 6520983700,
       5924182000, 6325786500, 5922986200, 5920388700, 5621485500,
       2322180400, 5320483400, 5920384000, 5925381200, 5320482000,
       5320481100, 5924182300, 7422787500, 5325181700, 5322681500,
       5920380400, 5925082400, 5322682800, 5320484600, 5320486900,
       5320481700, 6520681200, 6520681800, 1422783400, 1421586800,
       1824284700, 6525483800, 3222083900, 1824284000, 5321610100,
       5924486700, 5920386900, 5922387600, 2624410100, 5924187600])

# estimateMpnth select
estimateMonth = st.selectbox('Select your lot`s estimateMonth', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# estimateDay select
estimateDay = st.selectbox('Select your lot`s estimateDay', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])

# estimateYear radio
estimateYear = st.radio('What is your lot`s estimateYear', (2019,2020, 2021))

# daysDelta slider
daysDelta = st.slider('Please, select days delta for the lot', 68, 1160)

# daysRentPayDelta slider
daysRentPayDelta = st.slider('Please, select days rent pay delta for the lot', -522, 1396)

# daysRentPayDeltaSign radio
daysRentPayDeltaSign = st.radio('What is your lot`s daysRentPayDeltaSign', ("-","+"))

if daysRentPayDeltaSign == '+':
       daysRentPayDeltaSign = 1
else:
       daysRentPayDeltaSign = 0


# area_win slider
area_win = st.slider('Please, select area win for the lot', 0.1575, 10.276)

data = {
       'status':status,
       'pricePerOne': lot_pricePerOne,
       'estimatePrice': estimatePrice,
       'rentRate': rentRate,
       'rentalYield': rentalYield,
       'purpose': purpose,
       'koatuuLocation': koatuu,
       'ownerEdrpou': ownerEdrpou,
       'renterCompany': renterCompany,
       'renterEdrpou': renterEdrpou,
       'region_id': region_id,
       'estimateMonth': estimateMonth,
       'estimateDay': estimateDay,
       'estimateYear': estimateYear,
       'daysDelta': daysDelta,
       'daysRentPayDelta': daysRentPayDelta,
       'daysRentPayDeltaSign': daysRentPayDeltaSign,
       'area_win': area_win,
       'label': 0
}

# Creating a dataframe from the entered features
df2 = pd.DataFrame(data, index = [0])

# Merging dataset with the entered information for the futher processing
df_result = pd.concat([df,df2])

# Replacing outliers
outliers_replacement(df_result, 'estimatePrice', method='median')
outliers_replacement(df_result, 'daysDelta', method='median')
outliers_replacement(df_result, 'daysRentPayDelta', method='median')

#Numerical columns
num_col = ['estimatePrice', 'rentalYield', 'daysDelta', 'daysRentPayDelta', 'area_win']

#Categorical columns
cat_col = ['status', 'region_id', 'rentRate', 'purpose', 'koatuuLocation', 'ownerEdrpou', 'renterCompany', 'estimateMonth', 'estimateDay', 'estimateYear', 'daysRentPayDeltaSign']

label = 'label'

#Scaling numerical features
scaled_features = MinMaxScaler().fit_transform(df_result[num_col].values)
df_num = pd.DataFrame(scaled_features, index=df_result[num_col].index, columns=df_result[num_col].columns)
print(df_num.head())

#Categorical features processing

df_cat = df_result[cat_col]

# Label Encoding
for column in cat_col:
    df_cat[column] = df_cat[column].astype('category').cat.codes

# One-Hot Encoding
df_cat = pd.get_dummies(df_cat, columns=cat_col, dummy_na=False)


df_result = pd.concat([df_num, df_cat, df_result[label]], axis=1)

X = df_result[df_result[label] == 1]
X.drop(['label'], axis = 1, inplace = True)

lot = df_result[df_result[label] == 0]
lot.drop(['label'], axis = 1, inplace = True)

# Splitting data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=VAL_SIZE, shuffle = True, random_state=RANDOM_SEED)

#Catbost Regressor
model_catboost = CatBoostRegressor(iterations = 5000,                       
                          random_seed = RANDOM_SEED,
                          eval_metric='MAPE',
                          custom_metric=['RMSE', 'MAE'],
                          od_wait=500                          
                         )
model_catboost.fit(X_train, y_train,
         eval_set=(X_valid, y_valid),
         verbose_eval=100,
         use_best_model=True       
         )

test_predict_catboost = model_catboost.predict(X_valid)
print(f"The precision of the Catboosting Regressor by the MAPE metrics is: {(mape(y_valid, test_predict_catboost))*100:0.2f}%")

# Gradient Boosting Regressor with hyperparameters tuned
model_gbr = GradientBoostingRegressor(n_estimators=250, learning_rate= 0.1, max_depth=5, random_state=RANDOM_SEED)
model_gbr.fit(X_train, y_train)
y_pred_gbr = model_gbr.predict(X_valid)
print(f"The precision of the Gradient Bossting Regressor with hyperparameters tuned on MAPE metric is: {(mape(y_valid, y_pred_gbr))*100:0.2f}%")

# Doing blend prediction of the Catboost algorithm and Gradiend Boosting
blend_predict = (test_predict_catboost + y_pred_gbr) / 2
print(f"The precision of the blend of the best models by the MAPE metric is: {(mape(y_valid, blend_predict))*100:0.2f}%")

lot_pricePerOne_pred = (model_catboost.predict(lot) + model_gbr.predict(lot))/2

st.write("The lot's real price per ha is:", lot_pricePerOne)
st.write("The lot's predicted price per ha is:", int(lot_pricePerOne_pred[0]))

if lot_pricePerOne < int(lot_pricePerOne_pred[0]):
       st.success("This lot is cheaper than its market price. You should think about buying it")
else: 
       st.warning("This lot costs more than its market price. We don`t recommend buying it")