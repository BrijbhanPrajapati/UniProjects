{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 44,
      "metadata": {
        "id": "3kIvLR0WcZ9Z"
      },
      "outputs": [],
      "source": [
        "# Libreries\n",
        "import pandas as pd\n",
        "import nltk\n",
        "from nltk.tokenize import word_tokenize\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.sentiment.vader import SentimentIntensityAnalyzer\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.svm import LinearSVC"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "nltk.download('punkt')\n",
        "nltk.download('stopwords')\n",
        "nltk.download('vader_lexicon')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cG_vjBlZn95w",
        "outputId": "187f0223-a57f-47d5-8d41-1f4112f7a028"
      },
      "execution_count": 47,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Package punkt is already up-to-date!\n",
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n",
            "[nltk_data] Downloading package vader_lexicon to /root/nltk_data...\n",
            "[nltk_data]   Package vader_lexicon is already up-to-date!\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {},
          "execution_count": 47
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 46,
      "metadata": {
        "id": "VqFwlJTUcfvY"
      },
      "outputs": [],
      "source": [
        "\n",
        "# Loading the data here\n",
        "train_data = pd.read_csv(\"/content/drive/MyDrive/Projects/AAI Proj/archive/train.ft.txt\", sep='\\t', header=None, names=[\"review\", \"sentiment\"])\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#drive mount\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_XPAIz8L1CI_",
        "outputId": "e573eef2-b360-4f63-dea8-be170a6a477f"
      },
      "execution_count": 48,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# slice the data to load it fast.(beacause In the Dataset there is too much data)\n",
        "train_data = train_data[:40000]"
      ],
      "metadata": {
        "id": "AUpc1Ub8mxm0"
      },
      "execution_count": 49,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 50,
      "metadata": {
        "id": "N7Hhbh_sdDEH"
      },
      "outputs": [],
      "source": [
        "# Preprocess text (remove stopwords and lowercase)\n",
        "def preprocess_text(text):\n",
        "    tokens = word_tokenize(text.lower())\n",
        "\n",
        "    filtered_tokens = [token for token in tokens if token not in stop_words]\n",
        "    return ' '.join(filtered_tokens)\n",
        "stop_words = set(stopwords.words('english'))\n",
        "\n",
        "train_data['review'] = train_data['review'].apply(preprocess_text)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 51,
      "metadata": {
        "id": "D7zmZ9AzdE4P"
      },
      "outputs": [],
      "source": [
        "# Sentiment analysis using VADER\n",
        "vader = SentimentIntensityAnalyzer()\n",
        "\n",
        "def get_vader_sentiment(text):\n",
        "    scores = vader.polarity_scores(text)\n",
        "    sentiment = 'positive' if scores['pos'] > scores['neg'] else 'negative'\n",
        "    return sentiment\n",
        "\n",
        "train_data['vader_sentiment'] = train_data['review'].apply(get_vader_sentiment)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 42,
      "metadata": {
        "id": "2AOajUWRW3me"
      },
      "outputs": [],
      "source": [
        "# extraction of the Feature using TF-IDF\n",
        "vectorizer = TfidfVectorizer(max_features=2000)  # we can Adjust max_features if needed\n",
        "train_features = vectorizer.fit_transform(train_data['review'])\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Splitting the data into train and test sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(train_features, train_data['vader_sentiment'], test_size=0.2, random_state=42)\n",
        "\n",
        "# Training a Linear Support Vector Classifier\n",
        "classifier = LinearSVC()\n",
        "classifier.fit(X_train, y_train)\n",
        "\n",
        "# Predictions on the test set\n",
        "predictions = classifier.predict(X_test)\n",
        "\n",
        "# Evaluating the accuracy\n",
        "accuracy = accuracy_score(y_test, predictions)\n",
        "print(\"Accuracy:\", accuracy)\n",
        "\n",
        "# Print classification report\n",
        "print(\"\\nClassification Report:\")\n",
        "print(classification_report(y_test, predictions))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wPOWMJ8JnMZC",
        "outputId": "f8a66bfe-9308-421b-d618-08ccc620591f"
      },
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.873\n",
            "\n",
            "Classification Report:\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "    negative       0.77      0.68      0.72      1952\n",
            "    positive       0.90      0.93      0.92      6048\n",
            "\n",
            "    accuracy                           0.87      8000\n",
            "   macro avg       0.84      0.81      0.82      8000\n",
            "weighted avg       0.87      0.87      0.87      8000\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Counting positive and negative sentiments\n",
        "positive_count = (train_data['vader_sentiment'] == 'positive').sum()\n",
        "negative_count = (train_data['vader_sentiment'] == 'negative').sum()\n",
        "\n",
        "# Calculate Percentages\n",
        "total = len(train_data)\n",
        "positive_percentage = (positive_count / total) * 100\n",
        "negative_percentage = (negative_count / total) * 100\n",
        "\n",
        "# craeting pie chart\n",
        "labels = ['Positive', 'Negative']\n",
        "sizes = [positive_percentage, negative_percentage]\n",
        "colors = ['#ff9999','#66b3ff']\n",
        "plt.figure(figsize=(7,7))\n",
        "plt.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)\n",
        "plt.title('Sentiment Distribution')\n",
        "plt.axis('equal')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "Jlt2VK8quBFu",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 598
        },
        "outputId": "f3a06765-9bee-495c-c0b2-b4d5e05428dc"
      },
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 700x700 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjIAAAJFCAYAAADH6x0gAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABLzElEQVR4nO3dd3hUdaL/8c+k9wIEghCS0JsiXWpsCIorIpZVLOi6Xr3uevXqtT27ipW1Lld3Fdd7r4rib3UtWBBRBFSqiFJDh1BDCSQhIaTO+f1xSCAkgfTvOTPv1/PMQzKZTD4zoHz4tuOxLMsSAACACwWYDgAAAFBfFBkAAOBaFBkAAOBaFBkAAOBaFBkAAOBaFBkAAOBaFBkAAOBaFBkAAOBaFBkAAOBaFBnAASZNmqSUlBTTMYx7++235fF4lJGR0eQ/69T3PCMjQx6PRy+++GKT/2xJmjx5sjweT7P8LMCXUWTgd9asWaOrr75aycnJCgsLU7t27TRq1Ci9+uqrTfpz9+7dq8mTJ2vlypVN+nOaSkFBgSZPnqwFCxbU6vELFiyQx+OpuIWGhqpNmzY6//zz9eyzz+rgwYNGcjUnJ2cDfIWHay3BnyxevFgXXHCBOnTooFtuuUWJiYnatWuXli5dqq1bt2rLli1N9rN//vlnDRw4UG+99ZYmTZpU6WslJSXyer0KDQ1tsp/fUFlZWUpISNDjjz+uyZMnn/HxCxYs0AUXXKB77rlHAwcOVFlZmQ4ePKjFixfriy++UGxsrD788ENdeOGFFd9TVlamkpIShYaG1nq0oq65yp36nmdkZCg1NVUvvPCCHnjggVo/T32zlZaWqrS0VGFhYY3yswB/FWQ6ANCcnnnmGcXGxmr58uWKi4ur9LUDBw6YCSUpODjY2M9uaiNGjNDVV19d6b5Vq1bpkksu0YQJE5Senq62bdtKkgIDAxUYGNikeY4eParIyEjj73lQUJCCgvhfMNBQTC3Br2zdulW9evWqUmIkqXXr1lXue++999S/f3+Fh4erRYsW+u1vf6tdu3ZVesz555+v3r17Kz09XRdccIEiIiLUrl07Pf/88xWPWbBggQYOHChJuvXWWyumW95++21Jp1+v8fe//10dO3ZURESELrnkEu3atUuWZempp55S+/btFR4ernHjxunw4cNV8s+ePVsjRoxQZGSkoqOjNXbsWK1bt67SYyZNmqSoqCjt2bNHV155paKiopSQkKAHHnhAZWVlFXkSEhIkSU888URF/rqMgJysT58+mjp1qnJycvS3v/2t4v7q1sj8/PPPGj16tFq1aqXw8HClpqbqtttuq1Wu8te2detWXXbZZYqOjtbEiROrfc9P9te//lXJyckKDw9XWlqa1q5dW+nr559/vs4///wq33fyc54pW3VrZEpLS/XUU0+pU6dOCg0NVUpKih599FEVFRVVelxKSoouv/xyLVy4UIMGDVJYWJg6duyo6dOnV/+GAz6MIgO/kpycrBUrVlT5i6k6zzzzjG6++WZ16dJFL7/8su6991599913GjlypHJycio9Njs7W2PGjFGfPn300ksvqXv37nrooYc0e/ZsSVKPHj305JNPSpLuuOMOvfvuu3r33Xc1cuTI02aYMWOGXnvtNf3xj3/U/fffr++//17XXnut/vSnP+nrr7/WQw89pDvuuENffPFFlemQd999V2PHjlVUVJSee+45/fnPf1Z6erqGDx9eZTFtWVmZRo8erZYtW+rFF19UWlqaXnrpJf3jH/+QJCUkJOj111+XJI0fP74i/1VXXXXG97EmV199tcLDw/XNN9/U+JgDBw7okksuUUZGhh5++GG9+uqrmjhxopYuXVrrXKWlpRo9erRat26tF198URMmTDhtrunTp+uVV17R3XffrUceeURr167VhRdeqP3799fp9dXnPbv99tv12GOPqV+/fvrrX/+qtLQ0TZkyRb/97W+rPHbLli26+uqrNWrUKL300kuKj4/XpEmTqhRVwOdZgB/55ptvrMDAQCswMNAaMmSI9eCDD1pz5syxiouLKz0uIyPDCgwMtJ555plK969Zs8YKCgqqdH9aWpolyZo+fXrFfUVFRVZiYqI1YcKEivuWL19uSbLeeuutKrluueUWKzk5ueLz7du3W5KshIQEKycnp+L+Rx55xJJk9enTxyopKam4//rrr7dCQkKswsJCy7IsKy8vz4qLi7N+//vfV/o5+/bts2JjYyvdf8stt1iSrCeffLLSY/v27Wv179+/4vODBw9akqzHH3+8Sv7qzJ8/35Jk/etf/6rxMX369LHi4+MrPn/rrbcsSdb27dsty7KsTz/91JJkLV++vMbnOF2u8tf28MMPV/u16t7z8PBwa/fu3RX3L1u2zJJk3XfffRX3paWlWWlpaWd8ztNle/zxx62T/xe8cuVKS5J1++23V3rcAw88YEmy5s2bV3FfcnKyJcn64YcfKu47cOCAFRoaat1///1VfhbgyxiRgV8ZNWqUlixZoiuuuEKrVq3S888/r9GjR6tdu3b6/PPPKx73ySefyOv16tprr1VWVlbFLTExUV26dNH8+fMrPW9UVJRuvPHGis9DQkI0aNAgbdu2rUF5r7nmGsXGxlZ8PnjwYEnSjTfeWGl9xeDBg1VcXKw9e/ZIkr799lvl5OTo+uuvr5Q/MDBQgwcPrpJfku68885Kn48YMaLB+c8kKipKeXl5NX69fArwyy+/VElJSb1/zl133VXrx1555ZVq165dxeeDBg3S4MGD9dVXX9X759dG+fP/53/+Z6X777//fknSrFmzKt3fs2dPjRgxouLzhIQEdevWrcl/zwCnocjA7wwcOFCffPKJsrOz9dNPP+mRRx5RXl6err76aqWnp0uSNm/eLMuy1KVLFyUkJFS6rV+/vsrC4Pbt21dZ7xAfH6/s7OwGZe3QoUOlz8tLTVJSUrX3l/+8zZs3S5IuvPDCKvm/+eabKvnDwsIq1nM0Zv4zyc/PV3R0dI1fT0tL04QJE/TEE0+oVatWGjdunN56660qa0ZOJygoSO3bt6/147t06VLlvq5duzb52TY7duxQQECAOnfuXOn+xMRExcXFaceOHZXuP/XPhtQ8v2eA07BkHn4rJCREAwcO1MCBA9W1a1fdeuut+te//qXHH39cXq9XHo9Hs2fPrnYXTVRUVKXPa9ppYzXwdIOanvdMP8/r9Uqy18kkJiZWedypu2WaeqdQdUpKSrRp0yb17t27xsd4PB599NFHWrp0qb744gvNmTNHt912m1566SUtXbq0yu9DdUJDQxUQ0Lj/ZvN4PNX+3pYvjm7oc9dGU/2ZA9yGIgNIGjBggCQpMzNTktSpUydZlqXU1FR17dq1UX5Gc57i2qlTJ0n2TqyLL764UZ6zsfN/9NFHOnbsmEaPHn3Gx5533nk677zz9Mwzz+j999/XxIkT9c9//lO33357o+cqH8062aZNmyrtcIqPj692CufUUZO6ZEtOTpbX69XmzZvVo0ePivv379+vnJwcJScn1/q5AH/C1BL8yvz586v9F2v5+oRu3bpJkq666ioFBgbqiSeeqPJ4y7J06NChOv/syMhISaqy46kpjB49WjExMXr22WerXVtSn1N1IyIiJDVO/lWrVunee+9VfHy87r777hofl52dXeX9P/fccyWpYnqpMXNJ0syZMyvWGknSTz/9pGXLlunSSy+tuK9Tp07asGFDpfdx1apVWrRoUaXnqku2yy67TJI0derUSve//PLLkqSxY8fW6XUA/oIRGfiVP/7xjyooKND48ePVvXt3FRcXa/Hixfrggw+UkpKiW2+9VZL9F9XTTz+tRx55RBkZGbryyisVHR2t7du369NPP9Udd9xR59NfO3XqpLi4OE2bNk3R0dGKjIzU4MGDlZqa2uivMyYmRq+//rpuuukm9evXT7/97W+VkJCgnTt3atasWRo2bFil81tqIzw8XD179tQHH3ygrl27qkWLFurdu/dpp4Yk6ccff1RhYaHKysp06NAhLVq0SJ9//rliY2P16aefVjv1Ve6dd97Ra6+9pvHjx6tTp07Ky8vTm2++qZiYmIq/+OubqyadO3fW8OHDddddd6moqEhTp05Vy5Yt9eCDD1Y85rbbbtPLL7+s0aNH63e/+50OHDigadOmqVevXjpy5Ei93rM+ffrolltu0T/+8Q/l5OQoLS1NP/30k9555x1deeWVuuCCC+r1egCfZ2q7FGDC7Nmzrdtuu83q3r27FRUVZYWEhFidO3e2/vjHP1r79++v8viPP/7YGj58uBUZGWlFRkZa3bt3t+6++25r48aNFY9JS0uzevXqVeV7T92Ka1mW9dlnn1k9e/a0goKCKm3Frmkr8AsvvFDp+2va0ly+bfnUbcrz58+3Ro8ebcXGxlphYWFWp06drEmTJlk///xzpZyRkZFV8p+6PdiyLGvx4sVW//79rZCQkDNuxS7PWn4LDg62EhISrJEjR1rPPPOMdeDAgSrfc+r2619++cW6/vrrrQ4dOlihoaFW69atrcsvv7xS/tPlqum1lX+tpvf8pZdespKSkqzQ0FBrxIgR1qpVq6p8/3vvvWd17NjRCgkJsc4991xrzpw51f6e15Stuve3pKTEeuKJJ6zU1FQrODjYSkpKsh555JGKbfXlkpOTrbFjx1bJVNO2cMCXca0lAADgWqyRAQAArkWRAQAArkWRAQAArkWRAQAArkWRAQAArkWRAQAArkWRAQAArkWRAQAArkWRAQAArkWRAQAArkWRAQAArkWRAQAArkWRAQAArkWRAQAA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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "--------------------------------------------------------------------------"
      ],
      "metadata": {
        "id": "8tusUMtD0OwC"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 54,
      "metadata": {
        "id": "ia-GrXMUdJUH",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9c916ecf-d3bd-459c-dd33-14f0151e48ef"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "VADER Sentiment Distribution (Train Data):\n",
            "positive    30206\n",
            "negative     9794\n",
            "Name: vader_sentiment, dtype: int64\n"
          ]
        }
      ],
      "source": [
        "\n",
        "# Analyzing sentiments and distribution of train data using Vader\n",
        "vader_sentiment_counts = train_data['vader_sentiment'].value_counts()\n",
        "print(\"\\nVADER Sentiment Distribution (Train Data):\")\n",
        "print(vader_sentiment_counts)\n"
      ]
    }
  ],
  "metadata": {
    "accelerator": "TPU",
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}