from collections import namedtuple
from app.backbone.database.db_service import DbService
from app.backbone.entities.grouped_metrics import GroupedMetrics
from app.backbone.services.config_service import ConfigService
from app.backbone.services.operation_result import OperationResult
from app.backbone.services.utils import FtmoChallengeMetrics, MarginMetrics, calculate_margin_metrics, calculate_sharpe_ratio, calculate_stability_ratio, ftmo_simulator, get_portfolio_correlation_matrix, get_portfolio_equity_curve, get_portfolio_equity_differences, max_drawdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import jarque_bera, skew, kurtosis
import plotly.express as px


Metrics = namedtuple('Metrics',[
    'sharpe_ratio',
    'mean_returns',
    'std_returns',
    'stability_ratio',
    'return_',
    'dd', 
    'return_dd',
    'jarque_bera_stat', 
    'jarque_bera_p_value',
    'skew',
    'kurtosis'
])

class SystemPortfolioMetricsService():
    def __init__(self):
        self.db_service = DbService()
        self.config_service = ConfigService()


    def calculate_metrics_and_save(self, equity_curve, all_trades, system_id:int=None, portfolio_id:int=None):
        # Calculo las metricas del portfolio (return, dd, stability, negative hits, etc.)
        metrics = self.get_metrics(portfolio_equity_curve=equity_curve)
        challenge_metrics = self.get_challenge_metrics(equity_curve.Equity.values)
        margin_metrics = self.get_margin_metrics(all_trades, equity_curve)
        
        new_grouped_metrics = GroupedMetrics(
            SystemId=system_id,
            PortfolioId=portfolio_id,
            SharpeRatio=metrics.sharpe_ratio,
            StabilityRatio=metrics.stability_ratio,
            Return=metrics.return_,
            Drawdown=metrics.dd,
            RreturnDd=metrics.return_dd,
            JarqueBeraStat=metrics.jarque_bera_stat,
            JarqueBeraPValue=metrics.jarque_bera_p_value,
            Skew=metrics.skew,
            Kurtosis=metrics.kurtosis,
            MeanReturns=metrics.mean_returns,
            StdReturns=metrics.std_returns,
            PositiveHits=challenge_metrics.positive_hits,
            NegativeHits=challenge_metrics.negative_hits,
            SuccessRatio=challenge_metrics.success_ratio,
            MeanTimeToPositive=challenge_metrics.mean_time_to_positive,
            MeanTimeToNegative=challenge_metrics.mean_time_to_negative,
            StdTimeToPositive=challenge_metrics.std_time_to_positive,
            StdTimeToNegative=challenge_metrics.std_time_to_negative,
            MarginCalls = margin_metrics.margin_calls,
            StopOuts = margin_metrics.stop_outs
        )

        with self.db_service.get_database() as db:
            old_grouped_metrics = None

            if portfolio_id:
                old_grouped_metrics = self.db_service.get_by_filter(db, GroupedMetrics, PortfolioId=portfolio_id)
            
            elif system_id:
                old_grouped_metrics = self.db_service.get_by_filter(db, GroupedMetrics, SystemId=system_id)


            if old_grouped_metrics:
                new_grouped_metrics.Id = old_grouped_metrics.Id
                grouped_metrics = self.db_service.update(db, GroupedMetrics, new_grouped_metrics)

            else:
                grouped_metrics = self.db_service.create(db, new_grouped_metrics)
            
            return OperationResult(ok=True, message=None, item=grouped_metrics)

        return OperationResult(ok=False, message='There was an error', item=None)
    
    def get_full_equity_curve(self, equity_curves: dict) -> pd.Series:
        """Calcula la curva de equity del portafolio a partir de las curvas individuales."""
        initial_portfolio_cash = self.config_service.get_by_name(name='InitialCash').Value

        eq_curve = get_portfolio_equity_curve(all_bot_trades=equity_curves, initial_equity=float(initial_portfolio_cash))
        return eq_curve

    def get_full_equity_curve_plot(self, equity_curves: pd.DataFrame) -> str:
        # Crear una figura vacía
        fig = go.Figure()

        # Recorrer las curvas de equity de cada bot y agregarlas al gráfico
        for k, v in equity_curves.items():
            if k == 'portfolio':
                fig.add_trace(
                    go.Scatter(
                        x=v.index, 
                        y=v.Equity, 
                        mode="lines",
                        name=k
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=v.index, 
                        y=v.Equity, 
                        mode="lines+markers",
                        line_shape="hv",
                        name=k
                    )
                )

        # Actualizar los detalles del layout del gráfico
        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Equity",
            legend_title="Bots"
        )

        json_content = fig.to_json()
        
        return json_content
        
    def get_metrics(self, portfolio_equity_curve: pd.Series) -> Metrics:
        stability_ratio = calculate_stability_ratio(portfolio_equity_curve)
        return_ = ((portfolio_equity_curve.Equity.iloc[-1] - portfolio_equity_curve.Equity.iloc[0]) / portfolio_equity_curve.Equity.iloc[0]) * 100
        dd = np.abs(max_drawdown(portfolio_equity_curve, verbose=False))
        return_dd = return_ / dd
        
        returns = portfolio_equity_curve['Equity'].pct_change().dropna()  # Elimina NaN del primer valor
        mean_returns, std_returns = returns.mean(), returns.std()

        risk_free_rate = float(self.config_service.get_by_name('RiskFreeRate').Value)

        sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate=risk_free_rate)
        jb_stat, jb_p_value = jarque_bera(returns)
        jb_stat, jb_p_value = jb_stat, jb_p_value
        skew_value = skew(returns)
        kurtosis_value = kurtosis(returns, fisher=True)  # True para exceso sobre normal

        portfolio_metrics = Metrics(
            round(sharpe_ratio, 3),
            round(mean_returns, 3),
            round(std_returns, 3),
            round(stability_ratio, 3), 
            round(return_, 3), 
            round(dd, 3), 
            round(return_dd, 3),
            round(jb_stat, 3),
            round(jb_p_value, 3),
            round(skew_value, 3),
            round(kurtosis_value, 3)
        )
        return portfolio_metrics

    def get_challenge_metrics(self, portfolio_equity_curve: pd.Series) -> FtmoChallengeMetrics:
        initial_portfolio_cash = self.config_service.get_by_name(name='InitialCash').Value
        positive_hit_threshold = self.config_service.get_by_name(name='PositiveHitThreshold').Value
        negative_hit_threshold = self.config_service.get_by_name(name='NegativeHitThreshold').Value

        challenge_metrics = ftmo_simulator(
            portfolio_equity_curve, 
            initial_cash=float(initial_portfolio_cash),
            positive_hit_threshold=float(positive_hit_threshold),
            negative_hit_threshold=float(negative_hit_threshold),
        )

        return challenge_metrics
        
    def get_margin_metrics(self, all_trades:pd.DataFrame, portfolio_equity_curve:pd.Series) -> MarginMetrics:
        margin_metrics = calculate_margin_metrics(all_trades, portfolio_equity_curve)
        
        return margin_metrics

    def get_portfolio_correlation_matrix_plot(self, equity_curves, initial_cash):
        differences = get_portfolio_equity_differences(equity_curves, initial_cash)

        correlation_matrix = get_portfolio_correlation_matrix(differences=differences)

        fig = px.imshow(
            correlation_matrix.round(2).values,  
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            color_continuous_scale='Blues',
            zmin=-1, zmax=1
        )

        # Añadir anotaciones manualmente
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                fig.add_annotation(
                    text=str(round(correlation_matrix.iloc[i, j], 2)), 
                    x=j, y=i,
                    showarrow=False,
                    font=dict(color="black", size=12)
                )

        # Ajustar tamaño del gráfico y mejorar visibilidad de los nombres largos
        fig.update_layout(
            title="Correlation Matrix of Equity Curves",
            width=1000, height=750,  # Aumentar tamaño
            xaxis=dict(tickangle=-45, tickmode="array", tickvals=list(range(len(correlation_matrix.columns))), ticktext=correlation_matrix.columns),
            yaxis=dict(tickmode="array", tickvals=list(range(len(correlation_matrix.index))), ticktext=correlation_matrix.index)
        )

        # Convertir la figura a JSON
        graph_json = fig.to_json()
        
        return graph_json
