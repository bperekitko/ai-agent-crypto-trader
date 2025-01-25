from datetime import datetime


class Trade:
    def __init__(self, buyer, commission, commission_asset, trade_id, maker, order_id, price, qty, quote_qty, realized_pnl, side, position_side, symbol, time):
        self.buyer = buyer
        self.commission = commission
        self.commission_asset = commission_asset
        self.trade_id = trade_id
        self.maker = maker
        self.order_id = order_id
        self.price = price
        self.qty = qty
        self.quote_qty = quote_qty
        self.realized_pnl = realized_pnl
        self.side = side
        self.position_side = position_side
        self.symbol = symbol
        self.time = datetime.fromtimestamp(time // 1000)

    @classmethod
    def from_api_response(cls, msg):
        buyer = msg['buyer']
        commission = msg['commission']
        commission_asset = msg['commissionAsset']
        trade_id = msg['id']
        maker = msg['maker']
        order_id = msg['orderId']
        price = msg['price']
        qty = msg['qty']
        quote_qty = msg['quoteQty']
        realized_pnl = msg['realizedPnl']
        side = msg['side']
        position_side = msg['positionSide']
        symbol = msg['symbol']
        time = msg['time']
        return cls(buyer, commission, commission_asset, trade_id, maker, order_id, price, qty, quote_qty, realized_pnl, side, position_side, symbol, time)
