from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# التأكد من وجود مجلد templates
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app.template_folder = template_dir

print(f"📁 مسار القوابات: {app.template_folder}")
print(f"📁 هل المجلد موجود: {os.path.exists(template_dir)}")

if os.path.exists(template_dir):
    files = os.listdir(template_dir)
    print(f"📄 الملفات في templates: {files}")

# محاولة تحميل النموذج المدرب
try:
    model = joblib.load("profit_model_rf.pkl")
    logger.info("✅ تم تحميل النموذج بنجاح")
except Exception as e:
    logger.warning(f"⚠️ لم يتم العثور على النموذج: {e}")
    print("⚠️  لم يتم العثور على النموذج، سيتم استخدام نموذج محاكاة")
    model = None

def mock_predict(features):
    """
    دالة محاكاة للتنبؤ بالربح/الخسارة بناءً على المواصفات
    """
    # حساب الربح الأساسي
    revenue = features.get('unit_price', 0) * features.get('units_sold', 0)
    total_cost = (features.get('unit_cost', 0) * features.get('units_sold', 0) + 
                  features.get('advertising_cost', 0))
    
    # تطبيق الخصم
    discount_factor = 1 - (features.get('discount_rate', 0) / 100)
    net_revenue = revenue * discount_factor
    
    # الربح الصافي
    net_profit = net_revenue - total_cost
    
    # عوامل إضافية تؤثر على الربحية
    profit_factor = 1.0
    
    # تأثير تقييم العملاء
    rating = features.get('customer_rating', 3)
    if rating >= 4:
        profit_factor *= 1.2
    elif rating <= 2:
        profit_factor *= 0.8
    
    # تأثير الموسم
    season = features.get('season', 'Medium')
    if season == 'High':
        profit_factor *= 1.15
    elif season == 'Low':
        profit_factor *= 0.9
    
    # تأثير الطلب في السوق
    demand = features.get('market_demand', 'Medium')
    if demand == 'High':
        profit_factor *= 1.1
    elif demand == 'Low':
        profit_factor *= 0.85
    
    # تأثير المنافسة
    competition = features.get('competition_level', 'Medium')
    if competition == 'Low':
        profit_factor *= 1.1
    elif competition == 'High':
        profit_factor *= 0.9
    
    # تأثير عمر المنتج
    product_age = features.get('product_age_months', 6)
    if product_age < 3:
        profit_factor *= 1.05  # منتج جديد
    elif product_age > 12:
        profit_factor *= 0.95  # منتج قديم
    
    # تطبيق عوامل الربحية
    adjusted_profit = net_profit * profit_factor
    
    # التنبؤ بناءً على الربح المعدل
    if adjusted_profit > total_cost * 0.1:  # ربح أكثر من 10% من التكلفة
        return 'Profit', 0.85
    elif adjusted_profit < 0:  # خسارة
        return 'Loss', 0.75
    else:  # ربح ضئيل أو محايد
        return 'Neutral', 0.6

def preprocess_features(input_data):
    """
    معالجة وتحويل البيانات المدخلة إلى تنسيق مناسب للنموذج
    """
    processed = {}
    
    # البيانات الرقمية الأساسية
    processed['unit_cost'] = float(input_data.get('unit_cost', 0))
    processed['unit_price'] = float(input_data.get('unit_price', 0))
    processed['units_sold'] = int(input_data.get('units_sold', 0))
    processed['discount_rate'] = float(input_data.get('discount_rate', 0))
    processed['customer_rating'] = int(input_data.get('customer_rating', 3))
    processed['advertising_cost'] = float(input_data.get('advertising_cost', 0))
    processed['product_age_months'] = int(input_data.get('product_age_months', 0))
    
    # حساب الربح الأساسي
    revenue = processed['unit_price'] * processed['units_sold']
    total_cost = (processed['unit_cost'] * processed['units_sold'] + 
                  processed['advertising_cost'])
    discount_factor = 1 - (processed['discount_rate'] / 100)
    net_revenue = revenue * discount_factor
    net_profit = net_revenue - total_cost
    profit_margin = (net_profit / total_cost * 100) if total_cost > 0 else 0
    
    processed['net_profit'] = net_profit
    processed['profit_margin'] = profit_margin
    processed['revenue'] = revenue
    processed['total_cost'] = total_cost
    
    # ترميز فئة المنتج
    product_categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports', 'Books', 'Food', 'Other']
    category = input_data.get('product_category', 'Other')
    for cat in product_categories:
        processed[f'category_{cat}'] = 1 if category == cat else 0
    
    # ترميز الموسم
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    season = input_data.get('season', 'Winter')
    for s in seasons:
        processed[f'season_{s}'] = 1 if season == s else 0
    
    # ترميز مستوى المنافسة
    competition_levels = ['Low', 'Medium', 'High']
    competition = input_data.get('competition_level', 'Medium')
    for comp in competition_levels:
        processed[f'competition_{comp}'] = 1 if competition == comp else 0
    
    # ترميز الطلب في السوق
    demand_levels = ['Low', 'Medium', 'High']
    demand = input_data.get('market_demand', 'Medium')
    for dem in demand_levels:
        processed[f'demand_{dem}'] = 1 if demand == dem else 0
    
    return processed

def get_profit_analysis(prediction, probability, features):
    """
    تحليل النتيجة وتقديم توصيات
    """
    net_profit = features.get('net_profit', 0)
    profit_margin = features.get('profit_margin', 0)
    
    if prediction == 'Profit':
        if profit_margin > 20:
            return "🎉 أداء ممتاز! المنتج يحقق ربحية عالية. يمكن التفكير في زيادة الإنتاج."
        elif profit_margin > 10:
            return "👍 أداء جيد! المنتج مربح. حافظ على الاستراتيجية الحالية."
        else:
            return "✅ المنتج مربح ولكن يمكن تحسين الأداء. فكر في خفض التكاليف أو زيادة السعر."
    
    elif prediction == 'Loss':
        if profit_margin < -10:
            return "⚠️ خسارة كبيرة! يجب إعادة النظر في استراتيجية التسعير أو خفض التكاليف."
        else:
            return "📉 المنتج غير مربح حالياً. فكر في تحسين الجودة أو تغيير استراتيجية التسويق."
    
    else:  # Neutral
        return "📊 المنتج في حالة متوازنة. يمكن تحسين الربحية من خلال تحسين الكفاءة أو خفض التكاليف."

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"""
        <html>
            <body>
                <h1>خطأ في تحميل القالب</h1>
                <p>التفاصيل: {str(e)}</p>
                <p>مسار القوالب: {app.template_folder}</p>
                <p>هل المجلد موجود: {os.path.exists(app.template_folder)}</p>
            </body>
        </html>
        """

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'template_folder': app.template_folder,
        'template_folder_exists': os.path.exists(app.template_folder),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # الحصول على البيانات من الطلب
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        logger.info(f"📥 البيانات المستلمة: {data}")
        
        # معالجة البيانات
        processed_data = preprocess_features(data)
        
        logger.info(f"🔧 البيانات المعالجة: {processed_data}")
        
        # التنبؤ
        if model:
            # استخدام النموذج الحقيقي
            try:
                features_df = pd.DataFrame([processed_data])
                
                # التأكد من أن الأعمدة متوافقة مع ما تدرب عليه النموذج
                expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
                
                if expected_columns is not None:
                    # إعادة ترتيب الأعمدة لتتناسب مع النموذج
                    missing_cols = set(expected_columns) - set(features_df.columns)
                    for col in missing_cols:
                        features_df[col] = 0
                    features_df = features_df[expected_columns]
                
                # التنبؤ
                if hasattr(model, 'predict_proba'):
                    prediction_proba = model.predict_proba(features_df)[0]
                    prediction_class = model.predict(features_df)[0]
                    
                    # افتراض أن الفئة الثانية هي "Profit" إذا كان النموذج ثنائي التصنيف
                    if len(prediction_proba) == 2:
                        profit_probability = prediction_proba[1] * 100
                    else:
                        profit_probability = max(prediction_proba) * 100
                else:
                    prediction_class = model.predict(features_df)[0]
                    profit_probability = 75.0  # قيمة افتراضية
                
                prediction_map = {0: 'Loss', 1: 'Profit', 2: 'Neutral'}
                prediction = prediction_map.get(prediction_class, 'Neutral')
                
            except Exception as e:
                logger.error(f"❌ خطأ في استخدام النموذج: {e}")
                # العودة إلى المحاكاة في حالة الخطأ
                prediction, probability = mock_predict(processed_data)
                profit_probability = probability * 100
        else:
            # استخدام المحاكاة
            prediction, probability = mock_predict(processed_data)
            profit_probability = probability * 100
        
        # تحليل النتيجة
        analysis = get_profit_analysis(prediction, profit_probability, processed_data)
        
        response = {
            'status': 'success',
            'prediction': prediction,
            'prediction_probability': round(profit_probability, 2),
            'analysis': analysis,
            'net_profit': round(processed_data.get('net_profit', 0), 2),
            'profit_margin': round(processed_data.get('profit_margin', 0), 2),
            'model_used': 'real' if model else 'mock',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ التنبؤ الناجح: {prediction} ({profit_probability:.2f}%)")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ خطأ في التنبؤ: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error in prediction: {str(e)}'
        }), 400

@app.route('/test')
def test_page():
    """صفحة اختبار بسيطة للتأكد من عمل Flask"""
    return """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>اختبار Flask</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
            .success { color: green; }
            .error { color: red; }
        </style>
    </head>
    <body>
        <h1>✅ تطبيق Flask يعمل بنجاح!</h1>
        <p>إذا كنت ترى هذه الصفحة، فإن Flask يعمل بشكل صحيح.</p>
        <p>الآن يمكنك الانتقال إلى <a href="/">الصفحة الرئيسية</a></p>
        <p>أو فحص <a href="/health">صحة التطبيق</a></p>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("🚀 بدء تشغيل خادم Flask لتوقع الربح والخسارة...")
    print("📊 انتقل إلى http://localhost:5000 لرؤية الواجهة")
    print("🧪 انتقل إلى http://localhost:5000/test لفحص العمل")
    app.run(debug=True, host='0.0.0.0', port=5000)