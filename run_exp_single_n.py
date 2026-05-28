def run_experiment(
    X,
    y,
    T,
    sample_powers=range(3, 11),
    sigma=0.1,
    r=2,
    n_runs=1,
    device="cpu",
    max_epochs=200,
    arch="MLP",
    debug_summary_plot=True,
    arch_configs=None,
):

    if arch_configs is None:
        raise ValueError("arch_configs must be provided")

    model_builder = arch_configs[arch]["init"]

    sample_sizes = [2**p for p in sample_powers]

    results = {
        "clean": [],
        "noisy": [],
        "augmented": [],
    }

    for m in sample_sizes:

        assert m <= len(X)

        print("\n" + "=" * 60)
        print(f"Running experiments for m={m}")
        print("=" * 60)

        clean_accs, noisy_accs, aug_accs = [], [], []

        for run in range(n_runs):

            print(f"\nRun {run+1}/{n_runs}")

            # -----------------------------
            # SUBSAMPLE
            # -----------------------------
            idx = np.random.choice(len(X), size=m, replace=False)

            X_sub = X[idx]
            y_sub = y[idx]

            # -----------------------------
            # NOISE
            # -----------------------------
            noise = (
                np.random.normal(0, sigma, X_sub.shape)
                + 1j * np.random.normal(0, sigma, X_sub.shape)
            )

            X_noisy = X_sub + noise

            # =====================================================
            # CLEAN
            # =====================================================
            print("  Training CLEAN")

            train_loader, test_loader = build_dataloaders(
                X_sub,
                y_sub,
                augment=False
            )

            model = model_builder(T)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            history_clean = train_model(
                model,
                train_loader,
                test_loader,
                optimizer,
                max_epochs=max_epochs,
                device=device,
                verbose=False,
            )

            clean_accs.append(history_clean["best_val_acc"])

            # =====================================================
            # NOISY
            # =====================================================
            print("  Training NOISY")

            train_loader, test_loader = build_dataloaders(
                X_noisy,
                y_sub,
                augment=False
            )

            model = model_builder(T)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            history_noisy = train_model(
                model,
                train_loader,
                test_loader,
                optimizer,
                max_epochs=max_epochs,
                device=device,
                verbose=False,
            )

            noisy_accs.append(history_noisy["best_val_acc"])

            # =====================================================
            # AUGMENTED
            # =====================================================
            print("  Training AUGMENTED")

            train_loader, test_loader = build_dataloaders(
                X_noisy,
                y_sub,
                augment=True,
                r=r
            )

            model = model_builder(T)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            history_aug = train_model(
                model,
                train_loader,
                test_loader,
                optimizer,
                max_epochs=max_epochs,
                device=device,
                verbose=False,
            )

            aug_accs.append(history_aug["best_val_acc"])


        if DEBUG_SUMMARY_PLOT:

            fig, axs = plt.subplots(
                1,
                2,
                figsize=(14, 5)
            )

            # =====================================
            # LOSS
            # =====================================

            axs[0].plot(
                history_clean["train_losses"],
                label="Clean Train"
            )

            axs[0].plot(
                history_clean["val_losses"],
                linestyle="--",
                label="Clean Val"
            )

            axs[0].plot(
                history_noisy["train_losses"],
                label="Noisy Train"
            )

            axs[0].plot(
                history_noisy["val_losses"],
                linestyle="--",
                label="Noisy Val"
            )

            axs[0].plot(
                history_aug["train_losses"],
                label="Aug Train"
            )

            axs[0].plot(
                history_aug["val_losses"],
                linestyle="--",
                label="Aug Val"
            )

            axs[0].set_title(
                f"Loss Curves (m={m}, run={run+1})"
            )

            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")

            axs[0].legend()

            axs[0].grid(True)

            # =====================================
            # ACCURACY
            # =====================================

            axs[1].plot(
                history_clean["train_accs"],
                label="Clean Train"
            )

            axs[1].plot(
                history_clean["val_accs"],
                linestyle="--",
                label="Clean Val"
            )

            axs[1].plot(
                history_noisy["train_accs"],
                label="Noisy Train"
            )

            axs[1].plot(
                history_noisy["val_accs"],
                linestyle="--",
                label="Noisy Val"
            )

            axs[1].plot(
                history_aug["train_accs"],
                label="Aug Train"
            )

            axs[1].plot(
                history_aug["val_accs"],
                linestyle="--",
                label="Aug Val"
            )

            axs[1].set_title(
                f"Accuracy Curves (m={m}, run={run+1})"
            )

            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Accuracy")

            axs[1].legend()

            axs[1].grid(True)

            plt.suptitle(
                f"Training Diagnostics\n"
                f"Architecture=MLP, "
                f"Sigma={SIGMA}, "
                f"n=2^{int(np.log2(T))}"
            )

            plt.tight_layout()

            plt.show()

            plt.close()

        # =========================
        # AGGREGATION
        # =========================
        results["clean"].append(np.mean(clean_accs))
        results["noisy"].append(np.mean(noisy_accs))
        results["augmented"].append(np.mean(aug_accs))

        print(
            f"m={m} | "
            f"clean={np.mean(clean_accs):.3f}, "
            f"noisy={np.mean(noisy_accs):.3f}, "
            f"aug={np.mean(aug_accs):.3f}"
        )

    # =========================
    # SUMMARY PLOT
    # =========================
    if debug_summary_plot:

        plt.figure(figsize=(10, 6))

        plt.plot(sample_sizes, results["clean"], marker="o", label="Clean")
        plt.plot(sample_sizes, results["noisy"], marker="o", label="Noisy")
        plt.plot(sample_sizes, results["augmented"], marker="o", label="Augmented")

        plt.xscale("log", base=2)

        plt.xlabel("Sample Size m")
        plt.ylabel("Validation Accuracy")

        plt.title(
            f"Validation Accuracy vs Sample Size\n"
            f"Architecture={arch}, "
            fr"$\sigma={sigma}$, "
            rf"$r={r}$, "
            rf"$n=2^{int(np.log2(T))}$"
        )

        plt.grid(True)
        plt.legend()
        plt.show()

    return results