import numpy as np

def simulate_business_impact(baseline_auc, model_auc, num_orders=10000, avg_addon_value=5.0):
    """
    Simulate business impact.

    Args:
        baseline_auc (float): Baseline AUC.
        model_auc (float): Model AUC.
        num_orders (int): Number of orders.
        avg_addon_value (float): Average value of add-on.

    Returns:
        dict: Impact metrics.
    """
    # Assume lift in acceptance probability
    baseline_acceptance = 0.25  # Assume baseline acceptance rate
    lift = (model_auc - baseline_auc) * 0.5  # Rough estimate
    model_acceptance = baseline_acceptance * (1 + lift)

    additional_acceptances = (model_acceptance - baseline_acceptance) * num_orders
    revenue_impact = additional_acceptances * avg_addon_value

    # AOV lift: assume add-ons increase order value
    aov_lift = lift * 0.1  # 10% of lift

    # Cart-to-order ratio improvement: assume slight increase
    cart_to_order_improvement = lift * 0.05

    return {
        'baseline_acceptance': baseline_acceptance,
        'model_acceptance': model_acceptance,
        'additional_acceptances': additional_acceptances,
        'revenue_impact': revenue_impact,
        'aov_lift_percent': aov_lift * 100,
        'cart_to_order_improvement_percent': cart_to_order_improvement * 100
    }
