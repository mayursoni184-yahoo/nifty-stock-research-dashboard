import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="Nifty Pattern Research", page_icon="📈", layout="wide")

# Nifty 50 symbols. Yahoo Finance uses .NS for NSE-listed shares.
NIFTY = {
    "ADANIENT":"ADANIENT.NS","ADANIPORTS":"ADANIPORTS.NS","APOLLOHOSP":"APOLLOHOSP.NS","ASIANPAINT":"ASIANPAINT.NS","AXISBANK":"AXISBANK.NS","BAJAJ-AUTO":"BAJAJ-AUTO.NS","BAJFINANCE":"BAJFINANCE.NS","BAJAJFINSV":"BAJAJFINSV.NS","BEL":"BEL.NS","BHARTIARTL":"BHARTIARTL.NS","BPCL":"BPCL.NS","BRITANNIA":"BRITANNIA.NS","CIPLA":"CIPLA.NS","COALINDIA":"COALINDIA.NS","DRREDDY":"DRREDDY.NS","EICHERMOT":"EICHERMOT.NS","ETERNAL":"ETERNAL.NS","GRASIM":"GRASIM.NS","HCLTECH":"HCLTECH.NS","HDFCBANK":"HDFCBANK.NS","HDFCLIFE":"HDFCLIFE.NS","HINDALCO":"HINDALCO.NS","HINDUNILVR":"HINDUNILVR.NS","ICICIBANK":"ICICIBANK.NS","INDIGO":"INDIGO.NS","INDUSINDBK":"INDUSINDBK.NS","INFY":"INFY.NS","ITC":"ITC.NS","JIOFIN":"JIOFIN.NS","JSWSTEEL":"JSWSTEEL.NS","KOTAKBANK":"KOTAKBANK.NS","LT":"LT.NS","M&M":"M&M.NS","MARUTI":"MARUTI.NS","NESTLEIND":"NESTLEIND.NS","NTPC":"NTPC.NS","ONGC":"ONGC.NS","POWERGRID":"POWERGRID.NS","RELIANCE":"RELIANCE.NS","SBILIFE":"SBILIFE.NS","SBIN":"SBIN.NS","SHRIRAMFIN":"SHRIRAMFIN.NS","SUNPHARMA":"SUNPHARMA.NS","TATACONSUM":"TATACONSUM.NS","TATAMOTORS":"TATAMOTORS.NS","TATASTEEL":"TATASTEEL.NS","TCS":"TCS.NS","TECHM":"TECHM.NS","TITAN":"TITAN.NS","TRENT":"TRENT.NS","ULTRACEMCO":"ULTRACEMCO.NS","WIPRO":"WIPRO.NS"
}
PATTERNS=["Any","Cup & Handle","Head & Shoulders","Inverse Head & Shoulders","Double Top","Double Bottom","Rectangle Breakout","Ascending Triangle","Descending Triangle","Symmetrical Triangle","Reversal Bottom","Reversal Top"]

@st.cache_data(ttl=900, show_spinner=False)
def prices(ticker, period="5y"):
    d=yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if d is None or d.empty: return pd.DataFrame()
    d=d.rename(columns=str.lower)[["open","high","low","close","volume"]].dropna()
    d.index=pd.to_datetime(d.index).tz_localize(None)
    return d

@st.cache_data(ttl=43200, show_spinner=False)
def fundamentals(ticker):
    try:
        i=yf.Ticker(ticker).get_info()
        return {k:i.get(k) for k in ["marketCap","sector","industry","trailingPE","forwardPE","priceToBook","returnOnEquity","returnOnAssets","debtToEquity","currentRatio","profitMargins","operatingMargins","revenueGrowth","earningsGrowth","freeCashflow","operatingCashflow","dividendYield","fiftyTwoWeekHigh","fiftyTwoWeekLow","heldPercentInsiders"]}
    except Exception: return {}

def rs(d, rule):
    return d.resample(rule).agg(open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum")).dropna()

def pivots(a, kind, order=3):
    ans=[]
    for i in range(order,len(a)-order):
        w=a[i-order:i+order+1]
        if (kind=="high" and a[i]>=w.max()) or (kind=="low" and a[i]<=w.min()): ans.append(i)
    return ans

def base_result(name,status,df,level,direction,notes=""):
    c=float(df.close.iloc[-1]); vol=float(df.volume.iloc[-1]/max(df.volume.tail(21).iloc[:-1].mean(),1)-1)*100
    return {"pattern":name,"status":status,"date":df.index[-1],"level":float(level),"current":c,"return":(c/level-1)*100,"direction":direction,"volume":vol,"notes":notes}

def detect(df):
    if len(df)<35: return []
    out=[]; h=df.high.to_numpy(float); l=df.low.to_numpy(float); c=df.close.to_numpy(float)
    # double top/bottom using most recent two meaningful pivots
    ph=pivots(h,"high"); pl=pivots(l,"low")
    if len(ph)>=2:
        a,b=ph[-2],ph[-1]
        if b-a>=8 and abs(h[a]-h[b])/max(h[a],1)<.045:
            neck=float(l[a:b+1].min())
            if c[-1]<neck*.99: out.append(base_result("Double Top","Confirmed",df,neck,"Bearish","Breakdown below intervening support"))
            elif c[-1]<max(h[a],h[b])*1.01: out.append(base_result("Double Top","In progress",df,neck,"Bearish","Two similar highs; neckline not broken"))
    if len(pl)>=2:
        a,b=pl[-2],pl[-1]
        if b-a>=8 and abs(l[a]-l[b])/max(l[a],1)<.045:
            neck=float(h[a:b+1].max())
            if c[-1]>neck*1.01: out.append(base_result("Double Bottom","Confirmed",df,neck,"Bullish","Breakout above intervening resistance"))
            elif c[-1]>min(l[a],l[b])*.99: out.append(base_result("Double Bottom","In progress",df,neck,"Bullish","Two similar lows; neckline not broken"))
    # H&S from 3 latest pivot highs, inverse from 3 pivot lows
    if len(ph)>=3:
        a,b,z=ph[-3:]; shoulders=(h[a]+h[z])/2; neck=min(l[a:b+1].min(),l[b:z+1].min())
        if h[b]>shoulders*1.04 and abs(h[a]-h[z])/shoulders<.10:
            out.append(base_result("Head & Shoulders","Confirmed" if c[-1]<neck*.99 else "In progress",df,neck,"Bearish","Neckline confirmation required"))
    if len(pl)>=3:
        a,b,z=pl[-3:]; shoulders=(l[a]+l[z])/2; neck=max(h[a:b+1].max(),h[b:z+1].max())
        if l[b]<shoulders*.96 and abs(l[a]-l[z])/max(shoulders,1)<.10:
            out.append(base_result("Inverse Head & Shoulders","Confirmed" if c[-1]>neck*1.01 else "In progress",df,neck,"Bullish","Neckline confirmation required"))
    # Rectangle and triangles over 30 bars, scaled slopes
    w=30; x=np.arange(w); hh=h[-w:]; ll=l[-w:]; hs=np.polyfit(x,hh,1)[0]/max(hh.mean(),1); ls=np.polyfit(x,ll,1)[0]/max(ll.mean(),1)
    resistance=float(hh.max()); support=float(ll.min()); width=(resistance-support)/max(resistance,1)
    if width<.15:
        if abs(hs)<.001 and abs(ls)<.001: name="Rectangle Breakout"
        elif abs(hs)<.0007 and ls>.0007: name="Ascending Triangle"
        elif hs<-.0007 and abs(ls)<.0007: name="Descending Triangle"
        elif hs<-.0007 and ls>.0007: name="Symmetrical Triangle"
        else: name=None
        if name:
            if c[-1]>resistance*1.01: out.append(base_result(name,"Confirmed",df,resistance,"Bullish","Close above resistance"))
            elif c[-1]<support*.99: out.append(base_result(name,"Confirmed",df,support,"Bearish","Close below support"))
            else: out.append(base_result(name,"In progress",df,resistance if name!="Descending Triangle" else support,"Neutral","Consolidation remains inside boundaries"))
    # reversal candle and local location
    o=float(df.open.iloc[-1]); body=abs(c[-1]-o); rng=max(h[-1]-l[-1],1e-9); lower=min(o,c[-1])-l[-1]; upper=h[-1]-max(o,c[-1])
    prior_low=l[-11:-1].min(); prior_high=h[-11:-1].max()
    if lower/rng>.55 and body/rng<.35 and c[-1] <= prior_low*1.04: out.append(base_result("Reversal Bottom","Candidate",df,l[-1],"Bullish","Hammer-like candle near 10-bar low"))
    if upper/rng>.55 and body/rng<.35 and c[-1] >= prior_high*.96: out.append(base_result("Reversal Top","Candidate",df,h[-1],"Bearish","Shooting-star-like candle near 10-bar high"))
    return out

def levels(df):
    close=float(df.close.iloc[-1]); p1=pivots(df.low.to_numpy(),"low"); p2=pivots(df.high.to_numpy(),"high")
    lows=[float(df.low.iloc[i]) for i in p1[-8:] if df.low.iloc[i]<close]; highs=[float(df.high.iloc[i]) for i in p2[-8:] if df.high.iloc[i]>close]
    return (max(lows) if lows else float(df.low.tail(20).min()), min(highs) if highs else float(df.high.tail(20).max()))

def trend(df):
    c=float(df.close.iloc[-1]); ma20=df.close.rolling(20).mean().iloc[-1]; ma50=df.close.rolling(50).mean().iloc[-1]; ma200=df.close.rolling(200).mean().iloc[-1] if len(df)>=200 else np.nan
    if pd.notna(ma200) and c>ma20>ma50>ma200:return "Strong bullish"
    if c>ma20>ma50:return "Bullish"
    if c<ma20<ma50:return "Bearish"
    return "Neutral / consolidating"

def score(f):
    s=0; notes=[]
    def num(k): return f.get(k) if isinstance(f.get(k),(int,float)) else None
    roe=num("returnOnEquity"); debt=num("debtToEquity"); margin=num("profitMargins"); growth=num("revenueGrowth"); fcf=num("freeCashflow")
    if roe and roe>=.20:s+=2;notes.append("ROE ≥ 20%")
    elif roe and roe>=.15:s+=1;notes.append("ROE ≥ 15%")
    if debt is not None and debt<50:s+=2;notes.append("Low debt")
    elif debt is not None and debt<100:s+=1
    if margin and margin>=.15:s+=2;notes.append("Healthy profit margin")
    elif margin and margin>=.08:s+=1
    if growth and growth>=.12:s+=1;notes.append("Revenue growth")
    if fcf and fcf>0:s+=2;notes.append("Positive free cash flow")
    label="Excellent" if s>=7 else "Good" if s>=5 else "Average" if s>=3 else "Needs review"
    return s,label,notes

def chart(df, title, pats):
    d=df.tail(260).copy(); fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.76,.24],vertical_spacing=.03)
    fig.add_trace(go.Candlestick(x=d.index,open=d.open,high=d.high,low=d.low,close=d.close,name="Price"),1,1)
    fig.add_trace(go.Scatter(x=d.index,y=d.close.rolling(20).mean(),name="20 MA",line=dict(color="#1976d2")),1,1)
    fig.add_trace(go.Scatter(x=d.index,y=d.close.rolling(50).mean(),name="50 MA",line=dict(color="#ff9800")),1,1)
    fig.add_trace(go.Bar(x=d.index,y=d.volume,name="Volume",marker_color="#90a4ae"),2,1)
    for p in pats:
        fig.add_hline(y=p["level"],line_dash="dot",line_color="#8e24aa",annotation_text=p["pattern"],row=1,col=1)
    fig.update_layout(title=title,height=620,xaxis_rangeslider_visible=False,template="plotly_white",legend_orientation="h")
    return fig

def backtest(df, pattern):
    # Walk forward: use detector only on data available at each point; report last five confirmed events.
    events=[]; step=1
    for end in range(80,len(df)-21,step):
        found=[p for p in detect(df.iloc[:end]) if p["pattern"]==pattern and p["status"]=="Confirmed"]
        if found:
            p=found[-1]; day=df.index[end-1]
            if not events or events[-1]["date"]!=day:
                future=float(df.close.iloc[min(end+20,len(df)-1)]); ret=(future/p["current"]-1)*100
                signed=ret if p["direction"]=="Bullish" else -ret
                events.append({"date":day.date(),"breakout":p["current"],"20-bar return":signed,"success":signed>0})
    events=events[-5:]
    if not events:return None
    r=[e["20-bar return"] for e in events]
    return events,{"Occurrences":len(events),"Win rate":f"{100*np.mean([e['success'] for e in events]):.0f}%","Median return":f"{np.median(r):+.2f}%","Average return":f"{np.mean(r):+.2f}%"}

def fmt(v,pct=False):
    if v is None or not isinstance(v,(int,float)) or np.isnan(v):return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.2f}"

st.title("📊 Nifty Stock Research Dashboard")
st.caption("Latest available Yahoo Finance data • Pattern signals are rule-based research aids, not investment advice.")
with st.sidebar:
    mode=st.radio("Research mode",["Stock research", "Pattern scanner"])
    mincap=st.number_input("Minimum market cap (₹ crore)",value=3000,min_value=0,step=1000)
    if st.button("Refresh cached data"):
        st.cache_data.clear(); st.rerun()
    st.caption("Prices cache for 15 minutes; fundamentals cache for 12 hours.")

if mode=="Stock research":
    symbol=st.selectbox("Search Nifty stock",sorted(NIFTY),index=sorted(NIFTY).index("RELIANCE"))
    with st.spinner("Downloading latest market data..."):
        d=prices(NIFTY[symbol]); f=fundamentals(NIFTY[symbol])
    if d.empty: st.error("No price data returned. Please try Refresh cached data."); st.stop()
    cap=(f.get("marketCap") or 0)/1e7
    if cap and cap<mincap: st.warning(f"{symbol} is below your ₹{mincap:,.0f} crore filter.")
    support,resistance=levels(d); s,label,notes=score(f)
    a,b,c,dcol=st.columns(4); a.metric("Last close",f"₹{d.close.iloc[-1]:,.2f}"); b.metric("Market cap",f"₹{cap:,.0f} Cr" if cap else "—"); c.metric("Daily trend",trend(prices(NIFTY[symbol],"2y"))); dcol.metric("Fundamental score",f"{s}/9 · {label}")
    tabs=st.tabs(["Daily", "Weekly", "Monthly", "Fundamentals", "Historical evidence"])
    raw=prices(NIFTY[symbol],"5y")
    for tab,name,frame in zip(tabs,["Daily","Weekly","Monthly"],[raw,rs(raw,"W-FRI"),rs(raw,"ME")]):
        with tab:
            pp=detect(frame); st.subheader(name+" technical research")
            if pp:
                table=pd.DataFrame(pp)[["pattern","status","direction","date","level","current","return","volume","notes"]].rename(columns={"level":"Breakout / neckline","current":"Current price","return":"Return since level (%)","volume":"Volume vs 20 avg (%)"})
                st.dataframe(table,hide_index=True,use_container_width=True,column_config={"date":st.column_config.DatetimeColumn(format="YYYY-MM-DD"),"Breakout / neckline":st.column_config.NumberColumn(format="₹%.2f"),"Current price":st.column_config.NumberColumn(format="₹%.2f"),"Return since level (%)":st.column_config.NumberColumn(format="%.2f%%"),"Volume vs 20 avg (%)":st.column_config.NumberColumn(format="%.1f%%")})
            else: st.info("No supported confirmed/in-progress pattern currently detected on this timeframe.")
            x,y=levels(frame); q1,q2,q3=st.columns(3);q1.metric("Trend",trend(frame));q2.metric("Nearest support",f"₹{x:,.2f}");q3.metric("Nearest resistance",f"₹{y:,.2f}")
            st.plotly_chart(chart(frame,f"{symbol} — {name}",pp),use_container_width=True)
    with tabs[3]:
        st.subheader(f"Fundamental health: {label} ({s}/9)")
        st.write("Signals: "+(", ".join(notes) if notes else "insufficient comparable fields"))
        rows={"Sector":f.get("sector"),"Industry":f.get("industry"),"P/E":fmt(f.get("trailingPE")),"Forward P/E":fmt(f.get("forwardPE")),"Price/Book":fmt(f.get("priceToBook")),"ROE":fmt(f.get("returnOnEquity"),True),"ROA":fmt(f.get("returnOnAssets"),True),"Profit margin":fmt(f.get("profitMargins"),True),"Operating margin":fmt(f.get("operatingMargins"),True),"Revenue growth":fmt(f.get("revenueGrowth"),True),"Earnings growth":fmt(f.get("earningsGrowth"),True),"Debt/Equity":fmt(f.get("debtToEquity")),"Current ratio":fmt(f.get("currentRatio")),"Free cash flow":f"₹{(f.get('freeCashflow') or 0)/1e7:,.0f} Cr" if f.get("freeCashflow") else "—","Dividend yield":fmt(f.get("dividendYield"),True)}
        st.dataframe(pd.DataFrame(rows.items(),columns=["Metric","Value"]),hide_index=True,use_container_width=True)
        st.info("For banks/NBFCs, debt/equity and free-cash-flow comparisons are less meaningful. Add asset quality, CASA, NIM and capital-adequacy data from a dedicated Indian fundamentals provider before making banking decisions.")
    with tabs[4]:
        choices=sorted(set(p["pattern"] for p in detect(raw))) or PATTERNS[1:]
        pat=st.selectbox("Pattern to backtest",choices)
        if st.button("Run historical test (may take a few seconds)"):
            with st.spinner("Walking through historical price data..."):
                ans=backtest(raw,pat)
            if ans:
                ev,stats=ans; st.dataframe(pd.DataFrame(ev),hide_index=True,use_container_width=True); st.json(stats)
            else: st.info("No confirmed instances under these strict rules with enough forward data.")
else:
    pattern=st.selectbox("Pattern",PATTERNS); timeframe=st.selectbox("Timeframe",["Daily","Weekly","Monthly"]); status=st.selectbox("Status",["Any","Confirmed","In progress","Candidate"])
    if st.button("Scan Nifty universe",type="primary"):
        rows=[]; bar=st.progress(0)
        for n,(sym,tic) in enumerate(NIFTY.items(),1):
            try:
                f=fundamentals(tic); cap=(f.get("marketCap") or 0)/1e7
                if cap and cap<mincap: continue
                d=prices(tic,"5y"); frame=d if timeframe=="Daily" else rs(d,"W-FRI" if timeframe=="Weekly" else "ME")
                for p in detect(frame):
                    if (pattern=="Any" or p["pattern"]==pattern) and (status=="Any" or p["status"]==status): rows.append({"Stock":sym,"Market cap (Cr)":round(cap),"Pattern":p["pattern"],"Status":p["status"],"Direction":p["direction"],"Date":p["date"].date(),"Level":round(p["level"],2),"Current":round(p["current"],2),"Return %":round(p["return"],2),"Notes":p["notes"]})
            except Exception: pass
            bar.progress(n/len(NIFTY))
        bar.empty(); st.subheader(f"Matches: {len(rows)}")
        if rows: st.dataframe(pd.DataFrame(rows).sort_values(["Status","Return %"],ascending=[True,False]),hide_index=True,use_container_width=True)
        else: st.info("No matching current signals. Try a different timeframe, status, or pattern.")

st.divider(); st.caption("Data provided by Yahoo Finance. Verify prices, corporate actions, fundamentals and patterns independently. This application is for education/research only and is not investment advice.")
