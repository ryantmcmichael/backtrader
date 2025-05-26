import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
import random
from deap import base, creator, tools, algorithms
import multiprocessing
import warnings
from tqdm import tqdm
import copy
warnings.filterwarnings('ignore')

def generate_market_hours(start_date, end_date):
    """Generate DataFrame with market hours only."""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    market_times = []
    for date in dates:
        times = pd.date_range(
            start=date.replace(hour=6, minute=30),
            end=date.replace(hour=13, minute=0),
            freq='1min'
        )
        market_times.extend(times)

    return pd.DatetimeIndex(market_times)

def generate_stock_price(market_times, seed=None):
    """Generate synthetic stock price data with specified constraints."""
    if seed is not None:
        np.random.seed(seed)

    n_points = len(market_times)
    price = 100
    prices = [price]

    volatility = 0.001
    target_prob = 0.6

    for _ in range(n_points - 1):
        current_price = prices[-1]
        if current_price < 90 or current_price > 110:
            drift = 0
        else:
            drift = 0.0001 if np.random.random() < target_prob else -0.0001

        change = np.random.normal(drift, volatility)
        new_price = current_price * (1 + change)
        new_price = max(70, min(130, new_price))
        prices.append(new_price)

    df = pd.DataFrame(index=market_times)
    df['Open'] = prices
    df['Close'] = prices

    variation = np.random.uniform(0.997, 1.003, size=len(prices))
    df['High'] = df['Open'] * np.maximum(1, variation)
    df['Low'] = df['Open'] * np.minimum(1, variation)

    return df

def generate_market_signals(dates):
    """Generate daily market signals."""
    signals = pd.DataFrame(index=dates)

    for i in range(1, 7):
        signals[f'Signal_{i}'] = np.random.random(len(dates)) > 0.05

    return signals

def run_backtest_with_params(start_date, active_signals, profit_target, loss_target, n_periods=4):
    """Run backtest with optimizable parameters."""
    all_results = []
    period_returns = {}
    best_worst_stocks = {}
    all_stocks = {}

    for period in range(n_periods):
        period_start = start_date + timedelta(days=7*period)
        period_end = period_start + timedelta(days=8)

        market_times = generate_market_hours(period_start, period_end)
        business_days = pd.date_range(period_start, period_end, freq='B')
        market_signals = generate_market_signals(business_days)

        # Check only active signals
        first_thursday = period_start
        signals_check = []
        for i, is_active in enumerate(active_signals):
            if is_active:
                signals_check.append(market_signals.loc[first_thursday, f'Signal_{i+1}'])

        # If no signals are active, consider it True
        first_thursday_signals_ok = all(signals_check) if signals_check else True

        period_trades = []

        for stock_idx in range(4):
            stock_data = generate_stock_price(market_times, seed=period*100 + stock_idx)
            stock_name = f"Stock_{period}_{stock_idx+1}"
            all_stocks[stock_name] = stock_data

            if first_thursday_signals_ok:
                first_friday = first_thursday + timedelta(days=1)
                buy_minute = np.random.randint(2, 6)
                buy_time = datetime.combine(first_friday.date(),
                                         datetime.strptime(f"06:{32+buy_minute}", "%H:%M").time())

                buy_price = stock_data.loc[buy_time, 'Open']
                position = {'stock': stock_name,
                          'buy_time': buy_time,
                          'buy_price': buy_price,
                          'investment': 250}

                for current_time in market_times[market_times > buy_time]:
                    current_price = stock_data.loc[current_time, 'Open']
                    current_date = current_time.date()

                    exit_reason = None

                    if current_price >= buy_price * profit_target:
                        exit_reason = "Profit Target"

                    elif current_price <= buy_price * loss_target:
                        exit_reason = "Stop Loss"

                    elif current_date in market_signals.index:
                        # Check only active signals for exit
                        signals_check = []
                        for i, is_active in enumerate(active_signals):
                            if is_active:
                                signals_check.append(market_signals.loc[current_date, f'Signal_{i+1}'])
                        if signals_check and not all(signals_check):
                            exit_reason = "Signal False"

                    elif current_time.date() == period_end.date() and \
                         current_time.hour == 12:
                        exit_reason = "Period End"

                    if exit_reason:
                        position['sell_time'] = current_time
                        position['sell_price'] = current_price
                        position['exit_reason'] = exit_reason
                        position['return'] = (current_price - buy_price) / buy_price
                        period_trades.append(position)
                        break

        if period_trades:
            period_return = sum(trade['return'] * trade['investment']
                              for trade in period_trades)
            period_returns[period_end] = period_return

            sorted_trades = sorted(period_trades, key=lambda x: x['return'])
            best_worst_stocks[period] = {
                'worst': sorted_trades[0],
                'best': sorted_trades[-1]
            }

            all_results.extend(period_trades)

    return all_results, period_returns

# Set up the optimization problem
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Maximize returns, minimize drawdown
creator.create("Individual", list, fitness=creator.FitnessMulti)

def evaluate_strategy(individual):
    """Evaluate a single set of strategy parameters."""
    active_signals = individual[:6]
    profit_target = individual[6]
    loss_target = individual[7]

    start_date = pd.Timestamp('2024-03-07')
    results, period_returns = run_backtest_with_params(
        start_date=start_date,
        active_signals=active_signals,
        profit_target=profit_target,
        loss_target=loss_target
    )

    if not results:
        return -10000, 10000

    total_investment = sum(trade['investment'] for trade in results)
    total_return = sum(trade['return'] * trade['investment'] for trade in results)
    final_value = total_investment + total_return

    cumulative_returns = list(period_returns.values())
    max_drawdown = 0
    peak = float('-inf')

    for ret in cumulative_returns:
        if ret > peak:
            peak = ret
        drawdown = peak - ret
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return final_value, max_drawdown

def initialize_population():
    """Initialize a random individual."""
    signals = [random.randint(0, 1) for _ in range(6)]
    profit_target = random.uniform(1.01, 1.25)
    loss_target = random.uniform(0.80, 0.99)
    return signals + [profit_target, loss_target]

def mutate_parameters(individual):
    """Custom mutation function."""
    for i in range(6):
        if random.random() < 0.1:
            individual[i] = 1 - individual[i]

    if random.random() < 0.1:
        individual[6] += random.gauss(0, 0.01)
        individual[6] = max(1.01, min(1.25, individual[6]))

    if random.random() < 0.1:
        individual[7] += random.gauss(0, 0.01)
        individual[7] = max(0.80, min(0.99, individual[7]))

    return individual,

def setup_optimization():
    """Setup the genetic algorithm with parallel processing."""
    toolbox = base.Toolbox()

    # Register parallel evaluation
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    toolbox.register("individual", tools.initIterate, creator.Individual, initialize_population)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_strategy)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_parameters)
    toolbox.register("select", tools.selNSGA2)

    return toolbox, pool

def run_optimization(n_generations=50, population_size=100):
    """Run the optimization process with parallel processing and progress bars."""
    toolbox, pool = setup_optimization()

    print("Creating initial population...")
    population = toolbox.population(n=population_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print("\nStarting optimization...")
    gen_progress = tqdm(range(n_generations), desc="Generations", position=0)

    hof = tools.ParetoFront()

    # Evaluate initial population in parallel
    fitnesses = list(toolbox.map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(population)

    try:
        for gen in gen_progress:
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.3:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate invalid individuals in parallel
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            if invalid_ind:
                fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            population[:] = offspring
            hof.update(population)

            record = stats.compile(population)
            gen_progress.set_postfix({
                'Best Value': f"${max(ind.fitness.values[0] for ind in population):.2f}",
                'Min Drawdown': f"${min(ind.fitness.values[1] for ind in population):.2f}"
            })

    finally:
        pool.close()
        pool.join()

    return population, hof

def plot_pareto_front(population):
    """Plot the Pareto front of solutions."""
    plt.figure(figsize=(10, 6))

    final_values = []
    drawdowns = []

    for ind in population:
        if not ind.fitness.valid:
            continue
        final_values.append(ind.fitness.values[0])
        drawdowns.append(ind.fitness.values[1])

    plt.scatter(final_values, drawdowns, c='b', alpha=0.5)
    plt.xlabel('Final Portfolio Value ($)')
    plt.ylabel('Maximum Drawdown ($)')
    plt.title('Pareto Front of Solutions')
    plt.grid(True)
    plt.savefig('pareto_front.png')
    plt.close()

def print_best_solutions(population, hof):
    """Print details of the best solutions found."""
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)

    best_value = sorted(hof, key=lambda x: x.fitness.values[0], reverse=True)[:5]
    best_drawdown = sorted(hof, key=lambda x: x.fitness.values[1])[:5]

    print("\nTop 5 Solutions by Final Value:")
    for i, ind in enumerate(best_value, 1):
        print(f"\nSolution {i}:")
        print("Active Signals:", [f"Signal_{j+1}" for j, val in enumerate(ind[:6]) if val])
        print(f"Profit Target: {(ind[6]-1)*100:.1f}%")
        print(f"Loss Target: {(ind[7]-1)*100:.1f}%")
        print(f"Final Value: ${ind.fitness.values[0]:.2f}")
        print(f"Max Drawdown: ${ind.fitness.values[1]:.2f}")

    print("\nTop 5 Solutions by Minimum Drawdown:")
    for i, ind in enumerate(best_drawdown, 1):
        print(f"\nSolution {i}:")
        print("Active Signals:", [f"Signal_{j+1}" for j, val in enumerate(ind[:6]) if val])
        print(f"Profit Target: {(ind[6]-1)*100:.1f}%")
        print(f"Loss Target: {(ind[7]-1)*100:.1f}%")
        print(f"Final Value: ${ind.fitness.values[0]:.2f}")
        print(f"Max Drawdown: ${ind.fitness.values[1]:.2f}")

if __name__ == '__main__':
    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    print("\nStarting Trading Strategy Optimization")
    print("="*40)
    print(f"System Information:")
    print(f"- Using {num_cores} CPU cores for parallel processing")
    print("\nParameters:")
    print("- Population Size: 100")
    print("- Generations: 50")
    print("- Profit Target Range: 2% to 15%")
    print("- Loss Target Range: -10% to -1%")
    print("- Signals: 6 binary choices")
    print("="*40)

    try:
        population, hof = run_optimization()
        plot_pareto_front(population)
        print_best_solutions(population, hof)
        print("\nOptimization complete! Check 'pareto_front.png' for visualization.")
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Saving current results...")
        try:
            plot_pareto_front(population)
            print_best_solutions(population, hof)
        except:
            print("Could not save partial results.")
        print("\nExiting...")