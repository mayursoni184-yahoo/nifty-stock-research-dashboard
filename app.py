import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Nifty Total Market Research",page_icon="📊",layout="wide")
CSV="https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv"
PATS=["Any","Double Top","Double Bottom","Head & Shoulders","Inverse Head & Shoulders","Rectangle","Ascending Triangle","Descending Triangle","Symmetrical Triangle","Reversal Bottom","Reversal Top"]

@st.cache_data(ttl=21600)
def members():
    try:
        x=pd.read_csv(CSV); x=x[x.Series.astype(str).str.upper().eq("EQ")].copy();x.Symbol=x.Symbol.astype(str).str.upper().str.strip();x["Ticker"]=x.Symbol+".NS"
        return x[["Company Name","Industry","Symbol","Ticker"]].drop_duplicates("Symbol")
    except Exception:
        s=["RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","SBIN","LT","ITC"]
        return pd.DataFrame({"Company Name":s,"Industry":"Unavailable","Symbol":s,"Ticker":[z+".NS" for z in s]})
@st.cache_data(ttl=900)
def ohlcv(ticker,period="5y"):
    x=yf.Ticker(ticker).history(period=period,auto_adjust=True)
    if x is None or x.empty:return pd.DataFrame()
    x=x.rename(columns=str.lower)[["open","high","low","close","volume"]].dropna();x.index=pd.to_datetime(x.index).tz_localize(None);return x
@st.cache_data(ttl=43200)
def info(ticker):
    try:
        d=yf.Ticker(ticker).get_info();ks=["marketCap","sector","industry","trailingPE","forwardPE","priceToBook","returnOnEquity","returnOnAssets","debtToEquity","currentRatio","profitMargins","operatingMargins","revenueGrowth","earningsGrowth","freeCashflow","dividendYield","fiftyTwoWeekHigh","fiftyTwoWeekLow"]
        return {k:d.get(k) for k in ks}
    except:return {}
def tf(x,name):
    if name=="Daily":return x
    return x.resample("W-FRI" if name=="Weekly" else "ME").agg(open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum")).dropna()
def piv(a,high=True,n=3):
    return [i for i in range(n,len(a)-n) if (a[i]>=a[i-n:i+n+1].max() if high else a[i]<=a[i-n:i+n+1].min())]
def event(n,status,d,level,side,note):
    c=float(d.close.iloc[-1]);return {"Pattern":n,"Status":status,"Direction":side,"Date":d.index[-1],"Level":float(level),"Current":c,"Return %":(c/level-1)*100,"Volume %":(d.volume.iloc[-1]/max(d.volume.tail(21).iloc[:-1].mean(),1)-1)*100,"Notes":note}
def signals(d):
    if len(d)<40:return []
    h=d.high.to_numpy(float);l=d.low.to_numpy(float);c=d.close.to_numpy(float);ph=piv(h);pl=piv(l,False);o=[]
    if len(ph)>1:
        a,b=ph[-2:];neck=l[a:b+1].min()
        if b-a>7 and abs(h[a]-h[b])/h[a]<.045:o.append(event("Double Top","Confirmed" if c[-1]<neck*.99 else "In progress",d,neck,"Bearish","Two similar peaks; confirmation is below neckline"))
    if len(pl)>1:
        a,b=pl[-2:];neck=h[a:b+1].max()
        if b-a>7 and abs(l[a]-l[b])/max(l[a],1)<.045:o.append(event("Double Bottom","Confirmed" if c[-1]>neck*1.01 else "In progress",d,neck,"Bullish","Two similar troughs; confirmation is above neckline"))
    if len(ph)>2:
        a,b,z=ph[-3:];sh=(h[a]+h[z])/2;neck=min(l[a:b+1].min(),l[b:z+1].min())
        if h[b]>sh*1.04 and abs(h[a]-h[z])/sh<.1:o.append(event("Head & Shoulders","Confirmed" if c[-1]<neck*.99 else "In progress",d,neck,"Bearish","Confirmation is below neckline"))
    if len(pl)>2:
        a,b,z=pl[-3:];sh=(l[a]+l[z])/2;neck=max(h[a:b+1].max(),h[b:z+1].max())
        if l[b]<sh*.96 and abs(l[a]-l[z])/max(sh,1)<.1:o.append(event("Inverse Head & Shoulders","Confirmed" if c[-1]>neck*1.01 else "In progress",d,neck,"Bullish","Confirmation is above neckline"))
    w=30;hh=h[-w:];ll=l[-w:];hs=np.polyfit(range(w),hh,1)[0]/hh.mean();ls=np.polyfit(range(w),ll,1)[0]/ll.mean();r=hh.max();s=ll.min()
    if (r-s)/r<.15:
        name="Rectangle" if abs(hs)<.001 and abs(ls)<.001 else "Ascending Triangle" if abs(hs)<.0007 and ls>.0007 else "Descending Triangle" if hs<-.0007 and abs(ls)<.0007 else "Symmetrical Triangle" if hs<-.0007 and ls>.0007 else None
        if name:o.append(event(name,"Confirmed" if c[-1]>r*1.01 or c[-1]<s*.99 else "In progress",d,r if c[-1]>=s else s,"Bullish" if c[-1]>r else "Bearish" if c[-1]<s else "Neutral","Breakout requires a close outside the range"))
    op=d.open.iloc[-1];body=abs(c[-1]-op);rng=max(h[-1]-l[-1],1e-9)
    if (min(op,c[-1])-l[-1])/rng>.55 and body/rng<.35 and c[-1]<=l[-11:-1].min()*1.04:o.append(event("Reversal Bottom","Candidate",d,l[-1],"Bullish","Hammer-like candle near local low; await confirmation"))
    if (h[-1]-max(op,c[-1]))/rng>.55 and body/rng<.35 and c[-1]>=h[-11:-1].max()*.96:o.append(event("Reversal Top","Candidate",d,h[-1],"Bearish","Shooting-star-like candle near local high; await confirmation"))
    return o
def levels(d):
    c=d.close.iloc[-1];lo=[d.low.iloc[i] for i in piv(d.low.to_numpy(),False) if d.low.iloc[i]<c];hi=[d.high.iloc[i] for i in piv(d.high.to_numpy()) if d.high.iloc[i]>c]
    return max(lo[-8:]) if lo else d.low.tail(20).min(),min(hi[-8:]) if hi else d.high.tail(20).max()
def trend(d):
    c=d.close.iloc[-1];a=d.close.rolling(20).mean().iloc[-1];b=d.close.rolling(50).mean().iloc[-1];z=d.close.rolling(200).mean().iloc[-1] if len(d)>200 else np.nan
    return "Strong bullish" if pd.notna(z) and c>a>b>z else "Bullish" if c>a>b else "Bearish" if c<a<b else "Neutral / consolidation"
def fundscore(f):
    s=0;notes=[]
    for k,cut,points,label in [("returnOnEquity",.2,2,"ROE ≥20%"),("profitMargins",.15,2,"Profit margin ≥15%"),("revenueGrowth",.12,1,"Revenue growth ≥12%")]:
        if isinstance(f.get(k),(int,float)) and f[k]>=cut:s+=points;notes.append(label)
    if isinstance(f.get("debtToEquity"),(int,float)) and f["debtToEquity"]<50:s+=2;notes.append("Low debt")
    if isinstance(f.get("freeCashflow"),(int,float)) and f["freeCashflow"]>0:s+=2;notes.append("Positive FCF")
    return s,"Excellent" if s>=7 else "Good" if s>=5 else "Average" if s>=3 else "Needs review",notes
def fig(d,ps,title):
    x=d.tail(260);f=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.76,.24]);f.add_trace(go.Candlestick(x=x.index,open=x.open,high=x.high,low=x.low,close=x.close,name="Price"),1,1);f.add_trace(go.Scatter(x=x.index,y=x.close.rolling(20).mean(),name="20 MA"),1,1);f.add_trace(go.Scatter(x=x.index,y=x.close.rolling(50).mean(),name="50 MA"),1,1);f.add_trace(go.Bar(x=x.index,y=x.volume,name="Volume"),2,1)
    for p in ps:f.add_hline(y=p["Level"],line_dash="dot",annotation_text=p["Pattern"],row=1,col=1)
    f.update_layout(height=600,title=title,xaxis_rangeslider_visible=False,template="plotly_white",legend_orientation="h");return f

st.title("📊 Nifty Total Market (750) Research Dashboard")
st.caption("Uses the current Nifty Total Market constituent list and latest available Yahoo Finance data. Educational research only; not investment advice.")
with st.sidebar:
    mode=st.radio("Mode",["Stock research","Pattern scanner"]);minimum=st.number_input("Minimum market cap (₹ crore)",0,value=3000,step=1000)
    st.caption(f"Current index universe: {len(members())} EQ constituents")
    if st.button("Refresh all cached data"):st.cache_data.clear();st.rerun()
u=members()
if mode=="Stock research":
    syms=sorted(u.Symbol.tolist());sym=st.selectbox("Search a Nifty Total Market stock",syms,index=syms.index("RELIANCE") if "RELIANCE" in syms else 0);rec=u[u.Symbol.eq(sym)].iloc[0];d=ohlcv(rec.Ticker);f=info(rec.Ticker)
    if d.empty:st.error("No data returned. Refresh and try again.");st.stop()
    cap=(f.get("marketCap") or 0)/1e7;sc,lab,notes=fundscore(f);a,b,c,e=st.columns(4);a.metric("Last close",f"₹{d.close.iloc[-1]:,.2f}");b.metric("Market cap",f"₹{cap:,.0f} Cr" if cap else "Unavailable");c.metric("Trend",trend(d));e.metric("Fundamentals",f"{sc}/9 · {lab}")
    st.caption(f"{rec['Company Name']} • {rec.Industry}")
    tabs=st.tabs(["Daily","Weekly","Monthly","Fundamentals"])
    for tab,name in zip(tabs[:3],["Daily","Weekly","Monthly"]):
        with tab:
            q=tf(d,name);ps=signals(q);sup,res=levels(q);x,y,z=st.columns(3);x.metric("Trend",trend(q));y.metric("Support",f"₹{sup:,.2f}");z.metric("Resistance",f"₹{res:,.2f}")
            if ps:st.dataframe(pd.DataFrame(ps),hide_index=True,use_container_width=True,column_config={"Date":st.column_config.DatetimeColumn(format="YYYY-MM-DD"),"Level":st.column_config.NumberColumn(format="₹%.2f"),"Current":st.column_config.NumberColumn(format="₹%.2f"),"Return %":st.column_config.NumberColumn(format="%.2f%%"),"Volume %":st.column_config.NumberColumn(format="%.1f%%")})
            else:st.info("No supported active pattern found.")
            st.plotly_chart(fig(q,ps,f"{sym} — {name}"),use_container_width=True)
    with tabs[3]:
        st.subheader(f"Fundamental health: {lab} ({sc}/9)");st.write("Signals: "+(", ".join(notes) if notes else "insufficient fields"))
        def val(k,pc=False):
            v=f.get(k);return "—" if not isinstance(v,(int,float)) else f"{v*100:.1f}%" if pc else f"{v:.2f}"
        data={"Sector":f.get("sector"),"Industry":f.get("industry"),"P/E":val("trailingPE"),"Forward P/E":val("forwardPE"),"P/B":val("priceToBook"),"ROE":val("returnOnEquity",True),"ROA":val("returnOnAssets",True),"Profit margin":val("profitMargins",True),"Operating margin":val("operatingMargins",True),"Revenue growth":val("revenueGrowth",True),"Earnings growth":val("earningsGrowth",True),"Debt/Equity":val("debtToEquity"),"Current ratio":val("currentRatio"),"FCF":f"₹{f.get('freeCashflow',0)/1e7:,.0f} Cr" if f.get("freeCashflow") else "—","Dividend yield":val("dividendYield",True)}
        st.dataframe(pd.DataFrame(data.items(),columns=["Metric","Value"]),hide_index=True,use_container_width=True)
        st.info("For banks/NBFCs, use CASA, NIM, GNPA/NNPA, provision coverage and capital adequacy alongside these metrics.")
else:
    pat=st.selectbox("Pattern",PATS);time=st.selectbox("Timeframe",["Daily","Weekly","Monthly"]);stat=st.selectbox("Status",["Any","Confirmed","In progress","Candidate"])
    st.warning("A first 750-stock scan can take several minutes on free hosting. Results cache for 15 minutes. For dependable scheduled daily scans/alerts, use a database plus scheduled worker.")
    if st.button("Scan current Nifty 750",type="primary"):
        rows=[];bar=st.progress(0)
        for i,r in u.reset_index(drop=True).iterrows():
            try:
                d=tf(ohlcv(r.Ticker),time);m=[p for p in signals(d) if (pat=="Any" or p["Pattern"]==pat) and (stat=="Any" or p["Status"]==stat)]
                if m:
                    cap=(info(r.Ticker).get("marketCap") or 0)/1e7
                    if cap>=minimum:
                        for p in m:rows.append({"Stock":r.Symbol,"Company":r["Company Name"],"Industry":r.Industry,"Market cap Cr":round(cap),**p})
            except:pass
            bar.progress((i+1)/len(u))
        bar.empty();st.subheader(f"Matches: {len(rows)}")
        if rows:st.dataframe(pd.DataFrame(rows).sort_values(["Status","Return %"],ascending=[True,False]),hide_index=True,use_container_width=True)
        else:st.info("No matches. Try another pattern, timeframe, or status.")
st.divider();st.caption("Constituents: Nifty Indices. Prices/fundamentals: Yahoo Finance. Verify data, corporate actions and all signals independently.")
