# 🤖 AI-Powered CSV Visualizer

Transform your data into beautiful, meaningful insights with the power of artificial intelligence!

## 🌐 Try it now!

**🚀 [Launch the App](https://csv-viz.streamlit.app)** - No installation required!

*Just upload your CSV file and start discovering insights in seconds.*

## 🌟 What is this?

Imagine you have a spreadsheet full of numbers and text (like sales data, customer information, or survey results), but you're not sure what story your data is telling. This tool takes your spreadsheet file and uses artificial intelligence to:

- **Clean up your data** automatically (removes unnecessary columns, fixes formatting)
- **Understand what your data represents** (sales data, customer data, etc.)
- **Ask you what you want to discover** (gives you smart options to choose from)
- **Create beautiful charts and graphs** that answer your specific questions
- **Explain what each chart means** in plain English

Think of it as having a **personal data analyst** that works instantly and speaks your language!

## 🎯 Who is this for?

✅ **Business owners** who want to understand their sales, customers, or operations  
✅ **Students** working on projects with data  
✅ **Anyone curious** about patterns in their data  
✅ **People who find Excel overwhelming** but need insights  
✅ **Teams** that need quick data analysis without hiring analysts  

**No programming or technical skills required!**

## 🚀 What makes this special?

### 🧠 **Three AI Agents Working Together**
Think of it like having three smart assistants:

1. **🧠 Agent 1 - The Analyst**: Looks at your data and suggests what you might want to explore
2. **🎯 Agent 2 - The Strategist**: Decides the best way to visualize your specific questions  
3. **🖼 Agent 3 - The Designer**: Creates professional charts with explanations

### 💡 **Smart Options, Not Confusion**
Instead of guessing what charts to make, the AI:
- Analyzes your specific data
- Gives you 4 clear options like "Performance Analysis" or "Customer Behavior"
- Lets you ask custom questions in plain English
- Creates exactly what you need

### 📊 **Multiple Insights, Not Just One Chart**
You get a complete dashboard with:
- 3 different charts that work together
- Clear explanations of what each chart reveals
- Professional formatting you can share with others

## 🔧 How to use it (Step-by-step)

### **Step 1: Upload Your Data** 📂
- Save your spreadsheet as a CSV file (most programs can do this)
- Click "Upload a CSV file" and select your file
- The AI immediately starts understanding your data

### **Step 2: Review What AI Found** 🧠
- See a preview of your cleaned data
- Click "🚀 Start AI Analysis" 
- The AI tells you what your data represents and gives you options

### **Step 3: Choose What You Want to Explore** 🎯
Pick from options like:
- **Option 1**: "Performance Analysis" (revenue, trends, growth)
- **Option 2**: "Customer Behavior" (ratings, preferences, patterns)  
- **Option 3**: "Regional Comparison" (geographic differences)
- **Option 4**: "Product Analysis" (category performance)
- **Custom**: Ask your own question like "What affects my sales the most?"

### **Step 4: Get Your Insights** ✅
- The AI creates 3 professional charts
- Each chart answers your specific questions
- Every chart comes with a clear explanation
- Share the results with your team or use for decisions

### **Step 5: Export and Share** 📥
Download:
- **📄 Your cleaned data** (CSV file)
- **📋 Complete analysis report** (includes all insights)
- **📊 Professional dashboard** (HTML file you can open in any browser)

## 🎉 Example: What you might discover

**If you upload sales data, you might choose "Performance Analysis" and get:**

1. **📊 Chart 1**: "Sales Distribution by Month"  
   *"This chart shows your peak sales months are March and December, with a 40% increase during holidays"*

2. **🔍 Chart 2**: "Revenue vs Customer Ratings"  
   *"Higher customer ratings directly correlate with 25% more revenue per transaction"*

3. **📈 Chart 3**: "Regional Performance Comparison"  
   *"The North region outperforms others by 35%, suggesting successful local strategies"*

## 🛠 Setup (Simple Instructions)

### **What you need:**
- A computer with internet
- A CSV file (your spreadsheet saved as CSV)
- 5 minutes to set up

### **Setup steps:**

1. **Get the AI key** (free):
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with Google account
   - Click "Create API Key"
   - Copy the key (looks like: `abc123def456...`)

2. **Save your AI key**:
   - Create a new text file in the same folder as this project
   - Name it `.env` (just .env, no other name)
   - Inside, type: `GEMINI_API_KEY=paste_your_key_here`
   - Save the file

3. **Run the application**:
   - Open terminal/command prompt
   - Type: `streamlit run main.py`
   - Your browser will open with the application

## 💡 Tips for best results

### **Preparing your data:**
✅ **Use clear column names** (like "Sales Amount" instead of "Col1")  
✅ **Include dates** if you have time-based data  
✅ **Keep related data together** (don't split across multiple files)  
✅ **Remove empty rows** at the top or bottom  

### **Asking good questions:**
✅ **Be specific**: "How do customer ratings affect sales?" vs "Show me stuff"  
✅ **Focus on business goals**: "What drives my best performance?"  
✅ **Think about decisions**: "Which products should I focus on?"  

## 🎯 Sample data to try

The project includes sample files you can test with:
- **`sample_data.csv`**: Basic contact information (10 rows)
- **`sales_data.csv`**: Business sales data (20 rows) - perfect for testing all features

Try uploading these first to see how the AI works!

## 🆘 Common questions

**Q: What file types work?**  
A: Only CSV files. You can save any spreadsheet as CSV from Excel, Google Sheets, etc.

**Q: Is my data private?**  
A: Your data is processed through Google's AI, but not stored permanently. Read Google's privacy policy for details.

**Q: What if I don't like the AI's suggestions?**  
A: You can always ask custom questions or regenerate options. There's also manual exploration available.

**Q: Can I use this for large datasets?**  
A: Works best with datasets under 10,000 rows. Larger files may be slow.

**Q: Do I need to know programming?**  
A: Absolutely not! It's designed for non-technical users.

## 🎊 What people say

*"Finally, I can understand my sales data without learning Excel formulas!"*  
*"The AI asked exactly the right questions about my customer data."*  
*"I got insights in 5 minutes that would have taken me hours to figure out."*  
*"The explanations made everything clear - I could actually present this to my boss!"*

---

## 🚀 Ready to discover what your data is telling you?

**🌐 [Try it online now!](https://csv-viz.streamlit.app)**

Or run it locally:
1. Upload your CSV file
2. Let the AI analyze it
3. Choose what you want to explore  
4. Get instant, professional insights

**Transform your data into decisions in minutes, not hours!**

---

## 🔧 Technical Information

### **Tech Stack**
- **Frontend & Backend**: [Streamlit](https://streamlit.io/) - Python web framework
- **AI/LLM**: [Google Gemini 2.5 Flash](https://ai.google.dev/) - Latest AI model for analysis
- **Data Processing**: [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- **Visualizations**: [Plotly](https://plotly.com/python/) - Interactive charts and graphs
- **Environment**: [Python-dotenv](https://pypi.org/project/python-dotenv/) - Environment variable management

### **Architecture**
- **Multi-Agent System**: Three specialized AI agents working collaboratively
- **Session State Management**: Streamlit session state for smooth user experience
- **Smart Data Cleaning**: Automated preprocessing with intelligent column detection
- **Interactive UI**: Step-by-step guided analysis with user choice integration

### **Key Features**
- Automatic data type detection and conversion
- Smart column removal (IDs, constants, high cardinality)
- Missing value imputation (median for numeric, mode for categorical)
- Arrow serialization error handling
- Export capabilities (CSV, Markdown, HTML dashboard)

### **Development**
```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Run the application
streamlit run main.py
```

### **Contributing**
This project is designed to be user-friendly for non-technical users while maintaining clean, maintainable code for developers. Contributions welcome! 
