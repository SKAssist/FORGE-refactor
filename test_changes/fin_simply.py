def process_financial_data(transactions, categories=None, min_amount=0, 
                           date_range=None, include_pending=False):
    """
    Process financial transaction data for reporting and analysis.
    
    Args:
        transactions: List of transaction dictionaries
        categories: List of categories to include (None for all)
        min_amount: Minimum transaction amount to include
        date_range: Tuple of (start_date, end_date) or None for all
        include_pending: Whether to include pending transactions
        
    Returns:
        Dictionary with processed results including totals and statistics
    """
    # Validate input
    if not transactions:
        return {"error": "No transactions provided", "status": "failed"}
    
    # Initialize results
    filtered_transactions = []
    category_totals = {}
    
    # Process transactions
    for transaction in transactions:
        # Skip if amount is below minimum
        if transaction.get('amount', 0) < min_amount:
            continue
        
        # Skip if not in specified categories
        if categories and transaction.get('category') not in categories:
            continue
        
        # Skip if pending and not including pending
        if transaction.get('status') == 'pending' and not include_pending:
            continue
        
        # Skip if not in date range
        if date_range:
            start_date, end_date = date_range
            if start_date and transaction.get('date') < start_date:
                continue
            if end_date and transaction.get('date') > end_date:
                continue
                
        # Add to filtered transactions
        filtered_transactions.append(transaction)
        
        # Update category totals
        category = transaction.get('category', 'uncategorized')
        if category not in category_totals:
            category_totals[category] = {
                'count': 0,
                'total': 0,
                'min': float('inf'),
                'max': float('-inf')
            }
        
        amount = transaction.get('amount', 0)
        category_totals[category]['count'] += 1
        category_totals[category]['total'] += amount
        category_totals[category]['min'] = min(category_totals[category]['min'], amount)
        category_totals[category]['max'] = max(category_totals[category]['max'], amount)
    
    # Calculate grand totals
    grand_total = 0
    transaction_count = len(filtered_transactions)
    
    for category_data in category_totals.values():
        grand_total += category_data['total']
        # Fix any infinity values for empty categories
        if category_data['min'] == float('inf'):
            category_data['min'] = 0
        if category_data['max'] == float('-inf'):
            category_data['max'] = 0
    
    # Generate daily summaries
    daily_totals = {}
    for transaction in filtered_transactions:
        date = transaction.get('date')
        if date:
            if date not in daily_totals:
                daily_totals[date] = 0
            daily_totals[date] += transaction.get('amount', 0)
    
    # Sort transactions by date for better readability
    sorted_transactions = sorted(
        filtered_transactions, 
        key=lambda x: x.get('date', ''), 
        reverse=True
    )
    
    return {
        "status": "success",
        "transactions": sorted_transactions,
        "count": transaction_count,
        "grand_total": grand_total,
        "category_breakdown": category_totals,
        "daily_totals": daily_totals
    }