import stripe

stripe.api_key = os.getenv("STRIPE_API_KEY")

PLAN_PRICES = {"basic": 4900, "pro": 19900, "enterprise": 0} # valores em cents

def create_checkout_session(email, plan):
    return stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{
            'price_data': {
                'currency': 'eur',
                'product_data': {'name': f'{plan} plan'},
                'unit_amount': PLAN_PRICES[plan]
            },
            'quantity': 1,
        }],
        mode='payment',
        success_url='http://localhost:8501/?success=true',
        cancel_url='http://localhost:8501/?cancel=true',
        customer_email=email
    )
