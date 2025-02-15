import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

# Tải dataset
file_path = 'online_sales_dataset.csv'
df = pd.read_csv(file_path)

# Chuẩn bị các đặc trưng mới
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Revenue'] = df['Quantity'] * df['UnitPrice']
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['Month'] = df['InvoiceDate'].dt.month
df['Year'] = df['InvoiceDate'].dt.year
df['Quarter'] = df['InvoiceDate'].dt.quarter

# Tổng hợp dữ liệu theo danh mục, năm và quý
quarterly_demand_enhanced = df.groupby(['Year', 'Quarter', 'Category']).agg(
    TotalQuantity=('Quantity', 'sum'),
    TotalRevenue=('Revenue', 'sum'),
    AvgDiscount=('Discount', 'mean'),
    AvgShippingCost=('ShippingCost', 'mean'),
    TotalTransactions=('InvoiceNo', 'nunique')
).reset_index()

# Thêm các đặc trưng bổ sung
quarterly_demand_enhanced['QuarterIndex'] = (
    (quarterly_demand_enhanced['Year'] - quarterly_demand_enhanced['Year'].min()) * 4 
    + (quarterly_demand_enhanced['Quarter'] - 1)
)
quarterly_demand_enhanced['QuarterLabel'] = 'Q' + quarterly_demand_enhanced['Quarter'].astype(str) + '-' + quarterly_demand_enhanced['Year'].astype(str)

# Tạo biến giả cho 'Category'
demand_with_dummies_enhanced = pd.get_dummies(quarterly_demand_enhanced, columns=['Category'])

# Định nghĩa các đặc trưng và mục tiêu
X_enhanced = demand_with_dummies_enhanced.drop(columns=['Year', 'Quarter', 'TotalQuantity', 'QuarterLabel'])
y_enhanced = demand_with_dummies_enhanced['TotalQuantity']

# Chuẩn hóa các đặc trưng
scaler = StandardScaler()
X_enhanced_scaled = scaler.fit_transform(X_enhanced)

# Chia dataset
X_train_enhanced, X_test_enhanced, y_train_enhanced, y_test_enhanced = train_test_split(X_enhanced_scaled, y_enhanced, test_size=0.2, random_state=42)

# Tìm kiếm tham số tối ưu cho mô hình Random Forest
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='r2')
grid_search.fit(X_train_enhanced, y_train_enhanced)

# Huấn luyện mô hình Random Forest với tham số tối ưu
best_params = grid_search.best_params_
rf_model_enhanced = RandomForestRegressor(**best_params, random_state=42)
rf_model_enhanced.fit(X_train_enhanced, y_train_enhanced)

# Dự đoán trên tập test và tính R-squared
y_pred_enhanced = rf_model_enhanced.predict(X_test_enhanced)
r_squared = r2_score(y_test_enhanced, y_pred_enhanced)
rmse = mean_squared_error(y_test_enhanced, y_pred_enhanced, squared=False)

# Ứng dụng Streamlit
st.title("Hệ thống dự báo nhu cầu sản phẩm")

# Lựa chọn chức năng
functionality = st.sidebar.selectbox("Chọn chức năng:", ["Dự báo và trực quan hóa", "Xem lịch sử dự báo"])

history_file = 'forecast_history.csv'
if not os.path.exists(history_file):
    pd.DataFrame(columns=['Year', 'Quarter', 'Category', 'Prediction']).to_csv(history_file, index=False)

if functionality == "Dự báo và trực quan hóa":
    # Người dùng nhập thông tin để dự báo
    year = st.number_input("Nhập năm để dự báo:", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()) + 1, value=int(df['Year'].max()))
    quarter = st.selectbox("Chọn quý:", [1, 2, 3, 4])
    category = st.selectbox("Chọn danh mục sản phẩm:", df['Category'].unique())

    # Chuẩn bị dữ liệu để dự báo
    quarter_index = (year - df['Year'].min()) * 4 + (quarter - 1)
    new_data = pd.DataFrame({
        'QuarterIndex': [quarter_index],
        'TotalRevenue': [quarterly_demand_enhanced['TotalRevenue'].mean()],
        'AvgDiscount': [quarterly_demand_enhanced['AvgDiscount'].mean()],
        'AvgShippingCost': [quarterly_demand_enhanced['AvgShippingCost'].mean()],
        'TotalTransactions': [quarterly_demand_enhanced['TotalTransactions'].mean()]
    })

    # Thêm biến giả cho danh mục đã chọn
    for cat in demand_with_dummies_enhanced.columns:
        if "Category_" in cat:
            new_data[cat] = 1 if cat == f"Category_{category}" else 0

    # Khớp thứ tự cột
    new_data = new_data[X_enhanced.columns]

    # Hiển thị R-squared và RMSE của mô hình
    st.write("R-squared trên tập test:", r_squared)
    st.write("RMSE trên tập test:", rmse)

    # Lựa chọn phạm vi quý để hiển thị
    st.write("Chọn phạm vi quý để hiển thị:")
    quarter_start = st.selectbox("Chọn quý bắt đầu:", quarterly_demand_enhanced['QuarterLabel'].unique())
    quarter_end = st.selectbox("Chọn quý kết thúc:", quarterly_demand_enhanced['QuarterLabel'].unique())

    # Dự báo nhu cầu
    if st.button("Dự báo nhu cầu"):
        prediction = rf_model_enhanced.predict(new_data)[0]
        prediction = round(prediction,0)
        st.write(f"Số lượng dự báo cho {category} trong Q{quarter}, {year} là: {prediction:.2f}")

        # In kết quả dự báo ra console để kiểm tra
        print(f"Số lượng dự báo cho {category} trong Q{quarter}, {year} là: {prediction:.2f}")

        # Lưu lịch sử dự báo
        history = pd.read_csv(history_file) if os.path.getsize(history_file) > 0 else pd.DataFrame(columns=['Year', 'Quarter', 'Category', 'Prediction'])
        new_entry = pd.DataFrame({'Year': [year], 'Quarter': [quarter], 'Category': [category], 'Prediction': [prediction]})
        history = pd.concat([history, new_entry], ignore_index=True)
        history.to_csv(history_file, index=False)

        # Dữ liệu lịch sử cho danh mục đã chọn
        historical_data = quarterly_demand_enhanced[quarterly_demand_enhanced['Category'] == category].copy()

        # Lọc dữ liệu theo quý đã chọn
        quarter_start_index = historical_data[historical_data['QuarterLabel'] == quarter_start]['QuarterIndex'].values[0]
        quarter_end_index = historical_data[historical_data['QuarterLabel'] == quarter_end]['QuarterIndex'].values[0]
        filtered_data = historical_data[(historical_data['QuarterIndex'] >= quarter_start_index) & (historical_data['QuarterIndex'] <= quarter_end_index)]

        # Tạo hàng dữ liệu dự báo riêng biệt
        forecast_row = pd.DataFrame({
            'QuarterIndex': [quarter_index],
            'TotalQuantity': [prediction],
            'QuarterLabel': [f'Q{quarter}-{year}']
        })

        # Lấy điểm cuối cùng của dữ liệu lịch sử
        historical_data_filtered = filtered_data.copy()
        previous_quarter_point = historical_data_filtered.tail(1)

        # Kết hợp điểm cuối cùng của lịch sử với điểm dự báo để tạo đường dự báo riêng
        forecast_data = pd.concat([previous_quarter_point, forecast_row], ignore_index=True)

        # Biểu đồ 1: Chỉ hiển thị dữ liệu lịch sử
        fig1, ax1 = plt.subplots()
        ax1.plot(historical_data_filtered['QuarterIndex'], historical_data_filtered['TotalQuantity'], linestyle='-', color='blue', marker='o', label='Lịch sử nhu cầu')
        ax1.set_xticks(historical_data_filtered['QuarterIndex'])
        ax1.set_xticklabels(historical_data_filtered['QuarterLabel'], rotation=45, ha='right')
        ax1.set_xlabel('Quý')
        ax1.set_ylabel('Tổng số lượng')
        ax1.set_title(f"Lịch sử nhu cầu của {category}")
        ax1.legend()

        # Hiển thị biểu đồ 1 trong Streamlit
        st.pyplot(fig1)

        # Biểu đồ 2: Hiển thị cả dữ liệu lịch sử và dự báo
        fig2, ax2 = plt.subplots()

        # Vẽ đường lịch sử
        ax2.plot(historical_data_filtered['QuarterIndex'], historical_data_filtered['TotalQuantity'], linestyle='-', color='blue', marker='o', label='Lịch sử nhu cầu')

        # Vẽ đường dự báo nối từ quý trước đến quý dự báo
        ax2.plot(forecast_data['QuarterIndex'], forecast_data['TotalQuantity'], linestyle='--', color='red', marker='x', label='Dự báo nhu cầu')

        # Chú thích cho điểm dự báo
        ax2.annotate(f'{prediction:.2f}', (quarter_index, prediction), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='red')

        # Cập nhật nhãn trên trục x
        quarter_labels = list(historical_data_filtered['QuarterLabel']) + [f'Q{quarter}-{year}']
        ax2.set_xticks(list(historical_data_filtered['QuarterIndex']) + [quarter_index])
        ax2.set_xticklabels(quarter_labels, rotation=45, ha='right')

        # Thiết lập các thuộc tính cho biểu đồ
        ax2.set_xlabel('Quý')
        ax2.set_ylabel('Tổng số lượng')
        ax2.legend()
        ax2.set_title(f"Dự báo nhu cầu cho {category}")

        # Hiển thị biểu đồ 2 trong Streamlit
        st.pyplot(fig2)

elif functionality == "Xem lịch sử dự báo":
    # Xem lại lịch sử dự báo
    if st.button("Xem lịch sử dự báo"):
        history = pd.read_csv(history_file) if os.path.getsize(history_file) > 0 else pd.DataFrame(columns=['Year', 'Quarter', 'Category', 'Prediction'])
        st.write("Lịch sử dự báo:")
        st.dataframe(history)

    # Xóa lịch sử dự báo
    if st.button("Xóa lịch sử dự báo"):
        pd.DataFrame(columns=['Year', 'Quarter', 'Category', 'Prediction']).to_csv(history_file, index=False)
        st.write("Đã xóa lịch sử dự báo.")
