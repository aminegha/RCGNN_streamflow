# StreamPred

Predicting the spatiotemporal variation in streamflow, along with uncertainty quantification, enables informed decision-making for sustainable water resource management. While process-based hydrological models rely on physical laws, their simplifying assumptions can limit accuracy. Conversely, data-driven approaches require substantial training data and often produce results that conflict with physical laws. This project explores a constrained reasoning and learning (CRL) approach that integrates physical laws as logical constraints within deep neural networks. To address data scarcity, we propose a theoretically-grounded training method to enhance the generalization of deep models. For uncertainty quantification, we leverage the strengths of Gaussian processes (GPs) and deep temporal models by feeding the learned latent representation into a standard distance-based kernel. Experiments on real-world datasets validate the effectiveness of this combined CRL and GP approach, outperforming strong baseline methods.


### Environment Setup and Usage

- **Install the Environment:**

  Use the following command to create a Conda environment using the provided YAML file:

  ```bash
  conda env create -f env_streamflow_.yml
  ```

- **Run an example:**

  You can run for an example dataset using this command:

  ```bash
  python gnn_example.py
  ```

- **You can run a streamflow prediction model using this command:

  ```bash
  python gnn_tgds.py
  ```

- **Run the GP File:**

  For Uncertainty Qyantification, you can run the Gussian Process model using (You need to run a streamflow prediction model first):

  ```bash
  python gp_nn.py
  ```

- **Reference:**

  For more details, refer to the paper:
  Gharsallaoui, Mohammed Amine, et al. "Streamflow Prediction with Uncertainty Quantification for Water Management: A Constrained Reasoning and Learning Approach." arXiv preprint arXiv:2406.00133 (2024).


