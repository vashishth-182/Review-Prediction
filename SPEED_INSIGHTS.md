# 🚀 Vercel Speed Insights Integration

This document explains how Vercel Speed Insights has been integrated into this Streamlit application.

---

## 📊 What is Speed Insights?

Vercel Speed Insights allows you to track real-world performance metrics of your deployed application, including:
- **First Contentful Paint (FCP)**: Time until the first content is rendered
- **Largest Contentful Paint (LCP)**: Time until the largest content element is rendered
- **First Input Delay (FID)**: Time from user interaction to browser response
- **Cumulative Layout Shift (CLS)**: Visual stability of the page
- **Time to First Byte (TTFB)**: Server response time

---

## ✅ Implementation

Since this is a **Streamlit Python application** (not a JavaScript framework like Next.js or React), we've implemented Speed Insights using the HTML script injection method.

### Changes Made

#### 1. Modified `app.py`
Added Vercel Speed Insights tracking script using Streamlit's `components.html()` function:

```python
import streamlit.components.v1 as components

# Inject Vercel Speed Insights
speed_insights_script = """
<script>
    window.si = window.si || function () { (window.siq = window.siq || []).push(arguments); };
</script>
<script defer src="/_vercel/speed-insights/script.js"></script>
"""
components.html(speed_insights_script, height=0)
```

This injects the Speed Insights tracking script into the Streamlit app's HTML output without affecting the UI (height=0).

---

## 🔧 Setup on Vercel Dashboard

To enable Speed Insights for your deployed application:

### Step 1: Enable Speed Insights
1. Go to your [Vercel Dashboard](https://vercel.com/dashboard)
2. Select your **Review-Prediction** project
3. Navigate to the **Speed Insights** tab
4. Click **Enable**

> **Note:** After enabling, Speed Insights will add routes scoped at `/_vercel/speed-insights/*` after your next deployment.

### Step 2: Deploy Your App
Deploy your application to Vercel:

```bash
vercel deploy
```

Or connect your GitHub repository to Vercel for automatic deployments on every push to main.

### Step 3: Verify Installation
After deployment, verify that Speed Insights is working:

1. Visit your deployed app
2. Open browser DevTools (F12)
3. Check the Network tab for a request to `/_vercel/speed-insights/script.js`
4. If the script loads successfully, Speed Insights is active

### Step 4: View Your Data
Once users visit your site, view the performance data:

1. Go to your [Vercel Dashboard](https://vercel.com/dashboard)
2. Select your project
3. Click the **Speed Insights** tab
4. Explore real-time performance metrics

> **Note:** It may take a few hours to a day for data to appear after the first visits.

---

## 📈 What Gets Tracked?

Speed Insights automatically tracks:
- **Page Load Performance**: LCP, FCP, TTFB
- **Interactivity**: FID, INP (Interaction to Next Paint)
- **Visual Stability**: CLS
- **Route-specific metrics**: Performance data per page/route
- **Device and Browser breakdown**: Desktop vs Mobile performance

---

## 🔒 Privacy & Compliance

Vercel Speed Insights:
- Does **NOT** collect personally identifiable information (PII)
- Only tracks performance metrics and anonymous usage data
- Is GDPR and CCPA compliant
- Does not use cookies for tracking

[Learn more about Speed Insights privacy policy](https://vercel.com/docs/speed-insights/privacy-policy)

---

## 🎯 Performance Optimization Tips

Based on Speed Insights data, you can optimize your Streamlit app:

1. **Reduce Initial Load Time**
   - Optimize model file size (`lstm_model.h5`)
   - Use lazy loading for heavy imports
   - Consider model compression techniques

2. **Improve Interactivity**
   - Use `@st.cache_data` and `@st.cache_resource` decorators
   - Minimize re-runs on user input
   - Optimize prediction processing time

3. **Optimize for Mobile**
   - Test responsive design
   - Reduce resource sizes
   - Minimize JavaScript execution

---

## 📚 Additional Resources

- [Speed Insights Documentation](https://vercel.com/docs/speed-insights)
- [Speed Insights Metrics Explained](https://vercel.com/docs/speed-insights/metrics)
- [Speed Insights Pricing](https://vercel.com/docs/speed-insights/limits-and-pricing)
- [Streamlit Deployment on Vercel](https://docs.streamlit.io/deploy/streamlit-community-cloud)

---

## 🐛 Troubleshooting

### Script Not Loading?
- Verify Speed Insights is enabled in your Vercel project settings
- Check that you've deployed after enabling Speed Insights
- Ensure the app is deployed to Vercel (not Streamlit Cloud or other platforms)

### No Data in Dashboard?
- Wait 24-48 hours for initial data collection
- Verify the script loads in browser DevTools
- Ensure your app is receiving real user traffic

### Conflicts with Streamlit?
- The `components.html()` method is non-blocking and shouldn't affect app performance
- If issues occur, you can remove the Speed Insights code from `app.py`

---

## ✨ Benefits for This Project

For the Review Sentiment Predictor app, Speed Insights provides:
- **Model Loading Performance**: Track how quickly the LSTM model loads
- **Prediction Latency**: Monitor user interaction delays
- **Regional Performance**: Identify slow regions for your global users
- **Mobile vs Desktop**: Compare performance across device types

This data helps ensure a smooth user experience for anyone using your sentiment analysis tool!
