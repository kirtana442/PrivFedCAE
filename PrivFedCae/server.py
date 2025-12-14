# Import necessary modules from Flower framework
import flwr as fl
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from typing import List, Tuple


def server_fn(context: Context):
    """
    Flower server with malware detection metrics aggregation.

    Args:
        context: Flower Context object

    Returns:
        ServerAppComponents with FedAvg strategy
    """

    # Define custom FedAvg strategy that logs malware detection metrics
    class MalwareAwareFedAvg(fl.server.strategy.FedAvg):
        """
        Custom FedAvg strategy that aggregates and reports malware detection metrics from clients.
        """

        def aggregate_evaluate(self, server_round, results, failures):
            from pprint import pprint
            import json

            print(f"\n{'=' * 80}")
            print(f" [SERVER] Round {server_round} - Malware Detection Report")
            print(f"{'=' * 80}\n")

            malware_report = {
                "total_clients": len(results),
                "clients_with_detections": 0,
                "total_anomalies_detected": 0,
                "average_anomaly_rate": 0.0,
                "client_details": {},
            }

            total_anomaly_rate = 0.0

            if len(results) == 0:
                print("[SERVER] ⚠ No client evaluation results received.")
                print(f"\n[SERVER SUMMARY]")
                print(f"├─ Total clients: {malware_report['total_clients']}")
                print(f"├─ Clients with detections: {malware_report['clients_with_detections']}")
                print(f"├─ Total anomalies: {malware_report['total_anomalies_detected']}")
                print(f"└─ Average anomaly rate: {malware_report['average_anomaly_rate']:.2f}%")
                print(f"{'=' * 80}\n")

                aggregated_params, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
                aggregated_metrics.update(malware_report)
                return aggregated_params, aggregated_metrics

            # Process each client’s results
            for idx, (client_proxy, eval_res) in enumerate(results):
                try:
                    client_id = idx
                    metrics = getattr(eval_res, "metrics", None)
                    if not metrics:
                        continue

                    anomalies = int(metrics.get("malware_anomalies_detected", 0))
                    anomaly_rate = float(metrics.get("malware_anomaly_rate", 0.0))
                    mean_error = float(metrics.get("malware_mean_error", 0.0))
                    max_error = float(metrics.get("malware_max_error", 0.0))
                    threshold = float(metrics.get("detection_threshold", 0.0))

                    malware_report["client_details"][client_id] = {
                        "anomalies": anomalies,
                        "anomaly_rate": anomaly_rate,
                        "mean_error": mean_error,
                        "max_error": max_error,
                        "threshold": threshold,
                    }

                    if anomalies > 0:
                        malware_report["clients_with_detections"] += 1
                        print(f"  [Client {client_id}] MALWARE DETECTED")
                        print(f"    ├─ Anomalies: {anomalies}")
                        print(f"    ├─ Rate: {anomaly_rate:.2f}%")
                        print(f"    ├─ Mean Err: {mean_error:.6f}")
                        print(f"    ├─ Max Err: {max_error:.6f}")
                        print(f"    └─ Threshold: {threshold:.6f}\n")
                    else:
                        print(f"✓ [Client {client_id}] No anomalies detected\n")

                    total_anomaly_rate += anomaly_rate
                    malware_report["total_anomalies_detected"] += anomalies

                except Exception as e:
                    print(f"⚠ Exception while processing client {idx}: {e}")

            # Final summary
            malware_report["average_anomaly_rate"] = total_anomaly_rate / len(results)

            '''print(f"\n{'-' * 80}")
            print(f"[SERVER SUMMARY]")
            print(f"├─ Total clients: {malware_report['total_clients']}")
            print(f"├─ Clients with detections: {malware_report['clients_with_detections']}")
            print(f"├─ Total anomalies: {malware_report['total_anomalies_detected']}")
            print(f"└─ Avg anomaly rate: {malware_report['average_anomaly_rate']:.2f}%")
            print(f"{'-' * 80}\n")'''

            # Pretty-print JSON summary for history logs
            '''print("[SERVER] Detailed metrics per client:")
            print(json.dumps(malware_report["client_details"], indent=4))
            print(f"{'=' * 80}\n")'''

            aggregated_params, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
            aggregated_metrics.update(malware_report)
            return aggregated_params, aggregated_metrics

    config = ServerConfig(num_rounds=3)

    # Instantiate custom strategy with malware detection awareness
    strategy = MalwareAwareFedAvg(
        fraction_fit=1.0,              # Fraction of clients to sample for training
        fraction_evaluate=1.0,         # Fraction of clients to sample for evaluation
        min_fit_clients=1,             # Minimum clients required for training round
        min_evaluate_clients=1,        # Minimum clients required for evaluation round
        min_available_clients=1,       # Minimum clients that must be available to start
        evaluate_fn=None,              # No centralized evaluation function
    )

    # Return server components with the strategy
    return ServerAppComponents(strategy=strategy,config=config)


# Create the Flower server app
app = ServerApp(server_fn=server_fn)
