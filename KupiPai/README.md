# Land Lot Price Prediction

## Business Case
During the Soviet Perio all land in Ukraine belonged to the state. There were not private property at that time and all the land plots were in use of collective farms. 

After the Soviet Union collapsed the economy of Ukraine as independent state started moving towards the free market. The land lots that was previously in use of the collective farms were distributed between the members of particular collective farms as a shares and the members of the farms have become the owners of these land plots. 

However, since the day when Ukraine gained it Independence in 1991 and till July 2021 there was a moratorium in selling the land in Ukraine. Up until that time the most of the land plots were rented by big agriculture holdings and smaller farming enterprises which paid the land owners a monthly rent for using the plots for agricultural purposes.

Eventually in July 2021 the law was enacted that opened the land market in Ukraine. However, it was decided that the land market will be opening in few stages. At the first stage only individuals are allowed to buy and sell the land. Later, in few years, the land transactions will be allowed for commercial organization, and as the last step it will be allowed to foreigners. 

The opening of the land market in Ukraine caused a lot of buzz in the investors circles wich resulted in an emergence of third-party organizations and marketplaces which facilitate the land plots transactions.

## The purpose of the project
The purpose of the project is to build a model that predicts the price of a land lot based on its characteristics. Then knowing a real price of the particular land lot investors will be able to compare the real and predicted land lot`s price, which will help in making decision about buying the land lot.

## The project description
This project comprised several stages and corresponding files in the repository

1. **Data Collection.** The *requests* library was used to parse the data from https://kupipai.com.ua/ website which is an online marketplace for the land lot`s transaction. This allowed to collect all necessary features of available land lot's and save them in a csv file. **Check the file *KupiPaiParsing.ipynb***
2. **Exploratory Data Analysis.** A standard set of libraries such as Pandas, NumPy, Matplotlib, Seaborn was used to clean-up, enrich and process the data whic was parsed on the previous stage. The processed DataFrame was saved as a separate csv file for the future processing and modelling.**Check the file *KupiPaiEDA.ipynb***
3. **Data Modelling** On this stage the data as processed further(normalized, encoded) to prepare it to the modelling. A standard set of commonly used Machine Learning algorithms and their ensembles were tried. The effectiveness of the algorithms was assesed by MAPE metrics. As a result of this activity the best model was defined and tested. **Check the file *KupiPaiModel.ipynb***
4. **Working Prototype creation.** The *streamlit* library was used to build a working prototype wchich allows the user to use UI interface for the land lot price prediction.  **Check the file *kupipai.py***
5. **Production Deployment** An instance on AWS was launched with all necessary libraries and streamlit app was uploaded there. The TMUX was used to ensure that streamlit app is always up and running when AWS instance is running.
