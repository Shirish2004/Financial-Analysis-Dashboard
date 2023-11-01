#main page 
import streamlit as st 
import pandas as pd 
import yfinance as yf 
import matplotlib.pyplot as plt 
import seaborn as sn 
import numpy as np 
import time
import altair as alt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
# import pandas_profiling
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import datetime

#expanding the page width 
st. set_page_config(layout="wide")
#title of the page 
st.title("Stock and Portfolio Analysis")
stock = pd.read_csv("nasdaq_screener_1697553594138.csv")
stock_symbol = list(stock.Symbol)
#Functions required further 
#to display only the adjusted close price 
@st.cache_data
def combine_stock_data(_stocks, period, column_name='Adj Close'):
    stock_data = {}
    # Download and clean data for each stock
    for stock, ticker in _stocks.items():
        data = yf.download(stock, period=f"{period}mo")
        data = data[[column_name]]
        data = data.drop_duplicates()#drop duplicate values 
        stock_data[stock] = data
    # Merge dataframes based on stock names
    combined_data = pd.concat(stock_data, axis=1)
    combined_data.columns = combined_data.columns.droplevel(1)
    return combined_data
with st.sidebar:
    radio1 = st.radio(
        "***Select any one option***",
        ["**Stock Analysis**", "**Portfolio Risk Analysis**"],
        captions = ["Get a detailed overivew about a stock", "Compare stocks to know which one is riskier"])

if radio1 == "**Stock Analysis**":
    with st.sidebar:
        stock_for_analysis = st.selectbox("Select any stock ticker from the drop down menu :chart:",options = stock_symbol, 
                                        placeholder = "You can select only one stock")
        month = st.number_input("Input the number of months",min_value = 24, max_value = 84,step = 1, help = "You can enter values only upto 7 years back.",
            placeholder = "Enter the number of months")
    def data(symbol):
        ticker = yf.Ticker(symbol)
        data = yf.download(symbol,period=f"{month}mo")
        return ticker,data
    ticker,data = data(stock_for_analysis)
    tab1, tab2, tab3,tab4,tab5,tab6,tab7,tab8,tab9= st.tabs(["Stock Trade Values","Income Statement", "Balance Sheet", "Cashflow","Major Holders","Institutional Holders",
                                            "Call Option","Put Option","Mutual Fund Holder"])
    with tab1:
       profile =  ProfileReport(data, tsmode=True, title="Time-Series Analysis",correlations={"pearson": {"calculate": True},"spearman": {"calculate": True},"kendall": {"calculate": True},"phi_k": {"calculate": True}})
       st_profile_report(profile)
       export=profile.to_html()
       st.download_button(label="Download Report", data=export,key =1, file_name='report.html')
    with tab2:
        st.dataframe(ticker.income_stmt,use_container_width=True)
    with tab3:
        st.dataframe(ticker.balance_sheet,use_container_width=True)
    with tab4:
        st.dataframe(ticker.cashflow,use_container_width=True)
    with tab5:
        st.dataframe(ticker.major_holders,use_container_width=True)
    with tab6:
        st.dataframe(ticker.institutional_holders,use_container_width=True)
    with tab7:
        try:
            date = st.date_input("Select the date for your call option",datetime.date.today())
            opt = ticker.option_chain(f'{date}')
            profile =  ProfileReport(opt.calls,tsmode=True,correlations={"pearson": {"calculate": True},"spearman": {"calculate": True},"kendall": {"calculate": True},"phi_k": {"calculate": True}})
            st_profile_report(profile)
            export=profile.to_html()
            st.download_button(label="Download Report", data=export,key =2, file_name='report.html')
        except ValueError:
            st.write("#### ***The date you selected is either not an accepted expiry date or either it has passed please select another date***")
    with tab8:
        try:
            date = st.date_input("Select the date for your put option",datetime.date.today())
            opt = ticker.option_chain(f'{date}')
            profile =  ProfileReport(opt.puts,tsmode=True,correlations={"pearson": {"calculate": True},"spearman": {"calculate": True},"kendall": {"calculate": True},"phi_k": {"calculate": True}})
            st_profile_report(profile)
            export=profile.to_html()
            st.download_button(label="Download Report", data=export, key =3 ,file_name='report.html')
        except ValueError:
            st.write("#### ***The date you selected is either not an accepted expiry date or either it has passed please select another date***")
    with tab9:
        st.dataframe(ticker.mutualfund_holders,use_container_width=True)
      

    #Progress Bar for the download
    progress_text = "Downloading data :chart_with_upwards_trend:. Please wait...."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

elif radio1 == "**Portfolio Risk Analysis**":
    with st.sidebar:
        selected_stocks = st.multiselect("Select the stock tickers to find which one is riskier :chart:", options = stock_symbol, max_selections = 12,default = ["AAPL","AMZN","IBM","MSFT"],
                                        placeholder = "Choose stocks in your portfolio",
                                    )
    col1,col2 = st.columns(2)
    if len(selected_stocks) >=2:
        with col1:
            st.write("### **Your selected stocks are :** ", selected_stocks)
        
        with st.sidebar:
            st.write("## **Now select the period for which you want the analysis from the buttons, the minimum is 24 months and default**")
            month = st.number_input("Input the number of months",min_value = 24, max_value = 84,step = 1, help = "You can enter values only upto 7 years back.",
            placeholder = "Enter the number of months")
        stocks = {}
        for stock in selected_stocks:
            stocks[stock] = yf.Ticker(stock)
        #Progress bar for the download
        progress_text = "Downloading data :chart_with_upwards_trend:. Please wait...."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(2)
        my_bar.empty()
        stock_df = combine_stock_data(stocks,month,column_name = "Adj Close")
        stock_df = stock_df.dropna()
        tr_days_per_year = stock_df['AAPL'].groupby([stock_df['AAPL'].index.year]).agg('count')
        tr_days_per_year = pd.DataFrame([tr_days_per_year], index = ["All Stocks Combined"])
        with col1:
            st.write("### The dataframe displays the Adjusted Close price day wise for the stocks in your portfolio")
            st.dataframe(stock_df,
            use_container_width = True,
            )
            print(stock_df)
            st.write("### Number of trading days for your selected time period is")
            st.dataframe(tr_days_per_year)
            # Creating the animated plot
            @st.cache_data
            def simple_returns(df):
                simple_returns = df.apply(lambda x: x/x[0]-1)
                return simple_returns
            @st.cache_data
            def log_returns(df):
                log_returns = df.pct_change()
                return log_returns 
            log_returns = log_returns(stock_df)
            simple_returns = simple_returns(stock_df)
            st.write("#### The simple returns of each stock.")
            st.dataframe(simple_returns,use_container_width = True)
            st.write("#### The daily log returns of each stock.")
            st.dataframe(log_returns,use_container_width = True)
            #finding the annual percentage rate and its average
            @st.cache_data
            def APR_and_APRavg(log_re_df):
                apr = log_re_df.groupby([log_re_df.index.year]).agg('sum')
                apr_avg = apr.mean()
                apr_avg_df = pd.DataFrame(apr_avg, columns = ['Average APR']).T
                return apr, apr_avg,apr_avg_df
            APR,APR_avg,APR_avg_df = APR_and_APRavg(log_returns)
            st.write("### Your Annual Percentage Returns are")
            st.dataframe(APR,use_container_width = True)
            #calculating the annual percentage yield
            N = np.array(tr_days_per_year.T)
            @st.cache_data
            def APY_and_APYavg(APR,APR_avg,trading_days_per_year):
                N = np.array(trading_days_per_year.T)
                N_total = np.sum(N)
                APY = (1  + APR / N )**N - 1
                APY_avg = (1  + APR_avg /N_total  )**N_total - 1                
                APY_avg_df = pd.DataFrame(APY_avg, columns = ['Average APY']).T
                return APY,APY_avg,APY_avg_df
            APY,APY_avg,APY_avg_df = APY_and_APYavg(APR,APR_avg,tr_days_per_year)
            st.write("### Your Annual Percentage Yield are")
            st.dataframe(APY,use_container_width = True)
            # Now calculating the risk instrument that is standard deviation and variance 
            def std(log_returns_df):
                N = np.array(tr_days_per_year.T)
                STD       = log_returns.groupby([log_returns.index.year]).agg('std') * np.sqrt(N)
                STD_avg   = STD.mean()
                std       = log_returns.std()
                STD_avg_df = pd.DataFrame(STD_avg, columns = ['Average STD']).T
                return STD, STD_avg, std,STD_avg_df
            STD,STD_avg,std,STD_avg_df = std(log_returns)
            st.write("### Your Standard Deviation of Daily Returns are")
            st.dataframe(STD,use_container_width = True)
            fig, ax = plt.subplots(figsize = (16,12))
            ax.set_title(r"Standard Deviation of all stocks for all years")
            ax.set_facecolor((0, 0, 0))
            ax.grid(c = (0.25, 0.5, 0.25))
            ax.set_ylabel(r"Standard Deviation")
            ax.set_xlabel(r"Years")
            STD.plot(ax = plt.gca(),grid = True,linewidth = 3)

            for instr in STD:
                stds = STD[instr]
                years = list(STD.index)
                for year, std in zip(years, stds):
                    label = "%.3f"%std
                    plt.annotate(label, xy = (year, std), xytext=((-1)*50, 40),textcoords = 'offset points', ha = 'right', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0',color = 'white',lw = 3))
            with st.expander("**Expand to view plot of std. deviations of stocks**"):
                st.pyplot(fig)
            @st.cache_data
            def Var(STD_df):
                VAR = VAR = STD **2
                VAR_avg = VAR.mean()
                VAR_avg_df = pd.DataFrame(VAR_avg, columns = ['Average VAR']).T
                return VAR,VAR_avg,VAR_avg_df
            VAR,VAR_avg,VAR_avg_df = Var(STD)
            st.write("### Following is your variance dataframe")
            st.dataframe(VAR,use_container_width = True)
            c = [y + x for y, x in zip(APY_avg, STD_avg)]
            c = list(map(lambda x : x /max(c), c))
            s = list(map(lambda x : x * 600, c))
            # plot illustrating risk vs returns
            fig, ax = plt.subplots(figsize = (16,12))
            ax.set_title(r"Risk (STD) vs Return (APY) of all  stocks")
            ax.set_facecolor((0.95, 0.95, 0.99))
            ax.grid(c = (0, 0, 0))
            ax.set_xlabel(r"Standard Deviation")
            ax.set_ylabel(r"Annualized Percetaneg Yield (APY)")
            ax.scatter(STD_avg, APY_avg, s = s , c = c , cmap = "viridis", alpha = 0.8, edgecolors="grey", linewidth=3)
            ax.axhline(y = 0.0,xmin = 0 ,xmax = 5,c = "blue",linewidth = 3.5,zorder = 0,  linestyle = 'dashed')
            ax.axvline(x = 0.0,ymin = 0 ,ymax = 40,c = "blue",linewidth = 3.5,zorder = 0,  linestyle = 'dashed')
            for idx, instr in enumerate(list(STD.columns)):
                ax.annotate(instr, (STD_avg[idx] + 0.01, APY_avg[idx]))
            with st.expander("**Expand to view plot of Risk vs Returns**"):
                st.pyplot(fig)
            #getting the market values like risk free rates of T-bills and market returns of S&P 500
            @st.cache_data
            def risk_free_rate():
                risk_free = yf.download("^IRX",period = f"{month}mo")
                risk_free = risk_free["Adj Close"]
                risk_free = float(risk_free.tail(1))
                return risk_free
            risk_free = risk_free_rate()
            st.write(f"#### Most recent risk free rate of return in the trend is {risk_free}")
            @st.cache_data
            def market_return(log_returns_df):
                market = yf.download("^GSPC",period = f"{month}mo")
                market = market["Adj Close"]
                market = market.rename("^GSPC")
                market_log_returns = market.pct_change()
                log_returns_total  = pd.concat([log_returns,market_log_returns], axis = 1).dropna()
                return market,market_log_returns,log_returns_total
            market,market_log_returns,log_returns_total = market_return(log_returns)
            #RETURN
            APR_total         = log_returns_total.groupby([log_returns_total.index.year]).agg('sum')
            APR_avg_total     = APR_total.mean()
            APR_avg_market    = APR_avg_total['^GSPC']
            # RISK
            STD_total         = log_returns_total.groupby([log_returns_total.index.year]).agg('std') * np.sqrt(N)
            STD_avg_total     = STD_total.mean()
            STD_avg_market    = STD_avg_total['^GSPC']
            #finding correlation and r-squre
            corr = log_returns.corrwith(market_log_returns)
            r_squared = corr ** 2
            st.write("### The correlation of different instruments is viewed below:")
            st.dataframe(pd.DataFrame(r_squared, columns = ["R squared"]).T,use_container_width = True)
            #defining the CAPM model 
            @st.cache_data
            def CAPM():
                # 1 - Calculate average Risk Premium for every instrument  
                # [*]  _
                #     E[R] - R_f
                # [*]   __
                #     E[R_m] - R_f
                APR_premium        = APR_avg - risk_free
                APR_market_premium = APR_avg_market - risk_free
                # 2 - Calculate α, β
                beta  = corr *  STD_avg / STD_avg_market
                alpha = APR_premium - beta * APR_market_premium 
                return alpha, beta  
            alpha, beta = CAPM()
            instruments = list(log_returns.columns)
            def visualize_statistic(statistic, title, limit = 0):
                # configuration
                fig, ax = plt.subplots(figsize = (12,8))
                ax.set_facecolor((0, 0, 0))
                ax.grid(c = (0.25, 0.5, 0.25), axis = 'y')
                colors = sn.color_palette('Reds', n_colors = len(statistic))
                # visualize
                barlist = ax.bar(x = np.arange(len(statistic)), height =  statistic)
                for b, c in zip(barlist, colors):
                    b.set_color(c)
                ax.axhline(y = limit, xmin = -1 ,xmax = 1,c = "white",linewidth = 4,zorder = 0,  linestyle = 'dashed')

                # configure more
                for i, v in enumerate(statistic):
                    ax.text( i - 0.22,v + 0.01 , str(round(v,3)), color = 'white', fontweight='bold',size = 14,fontdict=None)
                plt.xticks(np.arange(len(statistic)), instruments)
                plt.title(r"{} for every instrument (i) against market (m) S&P500".format(title))
                plt.xlabel(r"Instrument")
                plt.ylabel(r"{} value".format(title))
                st.pyplot(plt.gcf())
            def visualize_model(alpha, beta, data, model):
                    fig, axs = plt.subplots(4,3, figsize = (14,10),  constrained_layout = True)
                    # fig.tight_layout()
                    idx = 0
                    R_m = data["^GSPC"]
                    del data["^GSPC"]
                    for a, b, instr in zip(alpha, beta, data):
                        i, j = int(idx / 3), idx % 3
                        axs[i, j].set_title("Model : {} fitted for '{}'".format(model, instr))
                        axs[i, j].set_facecolor((0,0,0))
                        axs[i, j].grid(c = (0.25,0.5,0.25))
                        axs[i, j].set_xlabel(r"Market (S&P500) log returns")
                        axs[i, j].set_ylabel(r"{} log returns".format(instr))
                        
                        R = data[instr]
                        y = a + b * R_m
                        axs[i, j].scatter(x = R_m, y = R, label = 'Returns'.format(instr))
                        axs[i, j].plot(R_m, y ,color = 'red', label = 'CAPM model')
                        idx += 1
                    #removing unused subplots 
                    for i in range(len(axs)):
                        for j in range(len(axs[i])):
                            if not axs[i][j].lines: axs[i][j].set_visible(False)
                    st.pyplot(plt.gcf())
            st.write("The alpha values for each stock ")
            alpha_df = pd.DataFrame(alpha)
            alpha_df.columns = ["Alpha"]
            st.dataframe(alpha_df,use_container_width = True)
            with st.expander("View the plot of alpha values"):
                visualize_statistic(alpha.values, "Alpha")
            st.write("The beta values for each stock ")
            beta_df = pd.DataFrame(beta)
            beta_df.columns = ["Beta"]
            st.dataframe(beta_df,use_container_width = True)
            with st.expander("View the plot of beta values"):
                visualize_statistic(beta.values, "Beta")
            st.write("## Visualize the model")
            with st.expander("Model Plots"):
                visualize_model(alpha/100, beta, data = log_returns_total.copy(), model = 'CAPM')
        with col2:
            st.markdown(f"### **Adjust Price of stocks over a period of {month} months**")
            chart1 = st.line_chart([])
            st.markdown(f"### **Daily Returns of stocks over a period of {month} months**")
            chart2 = st.line_chart([])
            st.markdown(f"### **Daily Log Returns of stocks over a period of {month} months**")
            chart3 = st.line_chart([])
            st.markdown(f"### **Annual Returns of stocks over a period of {month} months**")
            chart4 = st.line_chart([])
            st.markdown(f"### **Annual Percetnage Yields of stocks over a period of {month} months**")
            chart5 = st.line_chart([])
            st.markdown("### Alpha coefficient values for different stocks ")
            chart6 = st.empty()
            # max_height1 = max(alpha) * 1.2
            st.markdown("### Beta coefficient values for different stocks ")
            chart7 = st.empty()
            st.markdown("#### ***The labels in the above alpha and beta plots are stocks in the order selected in your selectbox***")
            # max_height2 = max(beta) * 1.2
            stock_names = [i for i in list(stock_df.columns)]
            stock_names = " ".join(stock_names)
            while True:
                for i in range(len(stock_df)):
                    # Update the chart with the current data
                    chart1.line_chart(stock_df.iloc[:i+1, 1:], use_container_width=True)
                    
                    chart2.line_chart(simple_returns.iloc[:i+1, 1:], use_container_width=True)
                    chart3.line_chart(log_returns.iloc[:i+1, 1:], use_container_width=True)
                    chart4.line_chart(APR.iloc[:i+1, 1:], use_container_width=True)
                    chart5.line_chart(APY.iloc[:i+1, 1:], use_container_width=True)
                    bar_heights6 = [val1 * (i / 100) for val1 in alpha]
                    chart6.bar_chart(bar_heights6,use_container_width=True)
                    bar_heights7 = [val2 * (i / 100) for val2 in beta]
                    chart7.bar_chart(bar_heights7, use_container_width=True)                    
                    # Sleep to control the animation speed
                    time.sleep(0.5)
                time.sleep(2)
        
    elif len(selected_stocks)<2:
        st.write("For this functionality to work you need to select atleast two stocks.")
    # performing the analysis over the stocks 

