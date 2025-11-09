"""
模型评估模块
负责评估模型性能并可视化结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import joblib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    def __init__(self, models_dir, data_path):
        self.models_dir = models_dir
        self.data_path = data_path
        self.models = {}
        self.results = {}

    def load_models(self):
        """加载训练好的模型"""
        print("正在加载模型...")

        model_files = {
            'Linear Regression': 'linear_regression.pkl',
            'Random Forest': 'random_forest.pkl',
            'XGBoost': 'xgboost.pkl'
        }

        for name, filename in model_files.items():
            filepath = os.path.join(self.models_dir, filename)
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"已加载: {name}")

        # 加载标准化器
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("已加载: Scaler")

    def load_data(self):
        """加载数据"""
        print("\n正在加载数据...")
        self.df = pd.read_csv(self.data_path)

        # 准备数据
        target = 'nat_demand'
        exclude_cols = ['datetime', 'nat_demand']

        feature_cols = [col for col in self.df.columns
                        if col not in exclude_cols and self.df[col].dtype in ['float64', 'int64']]

        X = self.df[feature_cols]
        y = self.df[target]

        # 划分训练集和测试集（与训练时保持一致）
        test_size = int(len(X) * 0.2)

        self.X_train = X.iloc[:-test_size]
        self.X_test = X.iloc[-test_size:]
        self.y_train = y.iloc[:-test_size]
        self.y_test = y.iloc[-test_size:]

        # 标准化（仅用于线性回归）
        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"测试集大小: {len(self.X_test)}")

    def evaluate_model(self, model, X_test, y_test, model_name):
        """评估单个模型"""
        # 预测
        y_pred = model.predict(X_test)

        # 计算评估指标
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # 保存结果
        self.results[model_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'predictions': y_pred
        }

        return mae, rmse, r2, y_pred

    def evaluate_all_models(self):
        """评估所有模型"""
        print("\n=== 模型评估 ===")

        for name, model in self.models.items():
            print(f"\n{name}:")

            # 线性回归使用标准化数据
            if name == 'Linear Regression':
                mae, rmse, r2, _ = self.evaluate_model(
                    model, self.X_test_scaled, self.y_test, name
                )
            else:
                mae, rmse, r2, _ = self.evaluate_model(
                    model, self.X_test, self.y_test, name
                )

            print(f"  MAE:  {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²:   {r2:.4f}")

    def plot_metrics_comparison(self, save_path):
        """绘制模型性能对比图"""
        print("\n正在生成性能对比图...")

        # 准备数据
        metrics_data = []
        for model_name, metrics in self.results.items():
            metrics_data.append({
                'Model': model_name,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2']
            })

        df_metrics = pd.DataFrame(metrics_data)

        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # MAE对比
        axes[0].bar(df_metrics['Model'], df_metrics['MAE'], color='skyblue')
        axes[0].set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('MAE')
        axes[0].tick_params(axis='x', rotation=45)

        # RMSE对比
        axes[1].bar(df_metrics['Model'], df_metrics['RMSE'], color='lightcoral')
        axes[1].set_title('Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('RMSE')
        axes[1].tick_params(axis='x', rotation=45)

        # R²对比
        axes[2].bar(df_metrics['Model'], df_metrics['R2'], color='lightgreen')
        axes[2].set_title('R² Score', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('R² Score')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"性能对比图已保存: {save_path}")
        plt.close()

    def plot_predictions(self, save_path, n_samples=500):
        """绘制预测结果对比图"""
        print("\n正在生成预测结果图...")

        fig, axes = plt.subplots(len(self.models), 1, figsize=(14, 4 * len(self.models)))

        if len(self.models) == 1:
            axes = [axes]

        y_test_array = self.y_test.values[:n_samples]

        for idx, (model_name, metrics) in enumerate(self.results.items()):
            y_pred = metrics['predictions'][:n_samples]

            axes[idx].plot(y_test_array, label='Actual', linewidth=2, alpha=0.7)
            axes[idx].plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
            axes[idx].set_title(f'{model_name} - Predictions vs Actual',
                                fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Time Steps')
            axes[idx].set_ylabel('Power Demand')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图已保存: {save_path}")
        plt.close()

    def plot_feature_importance(self, save_path):
        """绘制特征重要性图（仅适用于树模型）"""
        print("\n正在生成特征重要性图...")

        # 获取特征名
        feature_cols = [col for col in self.df.columns
                        if col not in ['datetime', 'nat_demand'] and
                        self.df[col].dtype in ['float64', 'int64']]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Random Forest特征重要性
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            importances_rf = rf_model.feature_importances_
            indices_rf = np.argsort(importances_rf)[-15:]  # 取前15个

            axes[0].barh(range(len(indices_rf)), importances_rf[indices_rf], color='skyblue')
            axes[0].set_yticks(range(len(indices_rf)))
            axes[0].set_yticklabels([feature_cols[i] for i in indices_rf], fontsize=9)
            axes[0].set_xlabel('Importance')
            axes[0].set_title('Random Forest - Top 15 Features', fontsize=12, fontweight='bold')

        # XGBoost特征重要性
        if 'XGBoost' in self.models:
            xgb_model = self.models['XGBoost']
            importances_xgb = xgb_model.feature_importances_
            indices_xgb = np.argsort(importances_xgb)[-15:]  # 取前15个

            axes[1].barh(range(len(indices_xgb)), importances_xgb[indices_xgb], color='lightcoral')
            axes[1].set_yticks(range(len(indices_xgb)))
            axes[1].set_yticklabels([feature_cols[i] for i in indices_xgb], fontsize=9)
            axes[1].set_xlabel('Importance')
            axes[1].set_title('XGBoost - Top 15 Features', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存: {save_path}")
        plt.close()

    def generate_report(self, save_path):
        """生成评估报告"""
        print("\n正在生成评估报告...")

        report = []
        report.append("=" * 60)
        report.append("电力负荷预测模型评估报告")
        report.append("=" * 60)
        report.append("")

        report.append(f"测试集大小: {len(self.y_test)}")
        report.append("")

        report.append("模型性能对比:")
        report.append("-" * 60)

        for model_name, metrics in self.results.items():
            report.append(f"\n{model_name}:")
            report.append(f"  MAE (平均绝对误差):     {metrics['MAE']:.2f}")
            report.append(f"  RMSE (均方根误差):      {metrics['RMSE']:.2f}")
            report.append(f"  R² Score (决定系数):    {metrics['R2']:.4f}")

        report.append("\n" + "=" * 60)
        report.append("结论:")
        report.append("-" * 60)

        # 找出最佳模型
        best_model = min(self.results.items(), key=lambda x: x[1]['RMSE'])
        report.append(f"\n最佳模型: {best_model[0]}")
        report.append(f"  RMSE: {best_model[1]['RMSE']:.2f}")
        report.append(f"  R²:   {best_model[1]['R2']:.4f}")

        report_text = "\n".join(report)

        # 保存报告
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"评估报告已保存: {save_path}")
        print("\n" + report_text)


def main():
    """主函数"""
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    data_path = os.path.join(base_dir, 'data', 'featured_data.csv')
    results_dir = os.path.join(base_dir, 'results')

    os.makedirs(results_dir, exist_ok=True)

    # 创建评估器
    evaluator = ModelEvaluator(models_dir, data_path)
    evaluator.load_models()
    evaluator.load_data()

    # 评估所有模型
    evaluator.evaluate_all_models()

    # 生成可视化结果
    evaluator.plot_metrics_comparison(os.path.join(results_dir, 'metrics_comparison.png'))
    evaluator.plot_predictions(os.path.join(results_dir, 'predictions.png'))
    evaluator.plot_feature_importance(os.path.join(results_dir, 'feature_importance.png'))

    # 生成报告
    evaluator.generate_report(os.path.join(results_dir, 'evaluation_report.txt'))

    print("\n模型评估完成！")


if __name__ == "__main__":
    main()