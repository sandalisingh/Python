from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import graphviz
from States import logging
from prettytable import PrettyTable
import tensorflow as tf

class DataVisualizer:
    @staticmethod
    def display_top_rows(df, num_rows=5, title=f"Top 5 rows of the dataset"):
        table = PrettyTable()
        table.field_names = list(df.columns)  # Extract column names
        rows = df.head(num_rows).values.tolist()
        table.add_rows(rows)  # Convert to list of lists
        table.title = title
        table.max_table_width = 100
        print(table)  

    @staticmethod
    def print_tensor_dict(title, data_dict):
        table = PrettyTable()
        table.title = title
        table.max_table_width = 120
        table.field_names = list(data_dict.keys())

        row_1 = []
        row_2 = []

        for key, tensor in data_dict.items():
            tensor_str = str(tensor)[:200] + "..." if len(str(tensor)) > 200 else str(tensor)
                
            row_1.append(tensor_str)

            if hasattr(tensor, "shape"):
                row_2.append(f"Shape: {tensor.shape}")
            else:
                row_2.append("Shape unknown")

        table.add_row(row_1, divider=True)
        table.add_row(row_2)

        print(table)

    @staticmethod
    def get_model_summary(model):
        model.summary()

    @staticmethod
    def get_arch_flowchat(model, file_name):
        plot_model(model, to_file=file_name, show_shapes=True, show_layer_names=True)
  
    @staticmethod
    def visualize_tensor_value_range(layer_index, layer_name, tensor):
        try:  
            plt.figure(figsize=(10, 5))
            plt.plot(tensor)
            plt.title(f'Layer {layer_index} - {layer_name}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging("error",f"Layer {layer_index} - {layer_name}: Error occurred during visualization - {str(e)}")

    @staticmethod
    def visualize_attention_heatmap(attention_weights, layer_name, layer_index):
        attention_weights = attention_weights.squeeze()  # Remove unnecessary dimensions
        plt.figure(figsize=(10, 5))
        plt.title(f'Attention Heatmap - Layer {layer_index}: {layer_name}')
        plt.imshow(attention_weights, cmap='viridis')
        plt.xlabel('Input Sequence')
        plt.ylabel('Output Sequence')
        plt.colorbar()
        plt.show()

    @staticmethod
    def plot_train_history(history, part_name_1, part_name_2, title):
        plt.figure(figsize=(12, 6))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history[part_name_1], label=part_name_1, color='red')
        plt.title(f'{title} : {part_name_1}')
        plt.xlabel('Epoch')
        plt.ylabel(part_name_1)
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history[part_name_2], label=part_name_2, color='blue')
        plt.title(f'{title} : {part_name_2}')
        plt.xlabel('Epoch')
        plt.ylabel(part_name_2)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'Plots/{title}_metrics.png')
        # plt.show()