# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Unified trainer for the cross-frequency LS benchmark.

One run = one immutable directory under runs/<name>/ containing config.json,
checkpoint.pt and train_log.csv.  Nothing is overwritten across runs.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P
import data as D
import models as M


def load_cache(folder, device):
    ds = D.CachedSet(folder)
    v = torch.from_numpy(np.concatenate([z["v"] for z in ds.data])).to(device)
    psi = torch.from_numpy(np.concatenate([z["psi"] for z in ds.data])).to(device)
    k = torch.from_numpy(np.concatenate([z["k"] for z in ds.data])).to(device)
    th = torch.from_numpy(np.concatenate([z["theta"] for z in ds.data])).to(device)
    f = torch.from_numpy(np.concatenate([z["f"] for z in ds.data])).to(device)
    return dict(v=v, psi=psi, k=k, th=th, f=f, max_res=ds.max_res())


def make_inputs(d, idx, emb="none", anchor="none"):
    v = d["v"][idx].float()
    psi_inc = P.incident_plane_wave(v.shape[-1], d["k"][idx], d["th"][idx],
                                    v.device).to(torch.complex64)
    f = d["f"][idx].float()
    if anchor == "none":
        return D.build_inputs(v, psi_inc, f, emb=emb), psi_inc, d["psi"][idx]
    born = D.anchor_terms(v, d["k"][idx], d["th"][idx])
    x = D.build_inputs(v, psi_inc, f, emb=emb, anchor=anchor, born=born)
    return x, psi_inc, d["psi"][idx], born


def rel_l2(pred, target):
    """Per-sample relative L2, returns vector (B,)."""
    num = torch.linalg.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    den = torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(1e-12)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="fno")
    ap.add_argument("--data", default=os.path.join(C.DATA, "train"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--proj", type=int, default=128)
    ap.add_argument("--norm", default="none")
    ap.add_argument("--f-emb", default="none", choices=["none", "ff"],
                    help="frequency input encoding: scalar channel (none) or "
                         "scalar + Fourier features (ff, C2 ablation)")
    ap.add_argument("--energy-lam", type=float, default=0.0,
                    help="C3 ablation: weight of the correction-energy penalty "
                         "lam * mean((log |u_hat|/|u|)^2); 0 = pure rel-L2")
    ap.add_argument("--anchor", default="none", choices=["none", "born"],
                    help="anchored residual operator: the network predicts the "
                         "residual on top of a.B1 + b.B2, the first two Born "
                         "corrections of the incident field (analytic at any "
                         "frequency, no label); a and b are two complex "
                         "coefficients trained on the band only")
    ap.add_argument("--ob-energy-lam", type=float, default=0.0,
                    help="C3b ablation: weight of an OUT-OF-BAND energy penalty "
                         "anchored on the Born correction field; acts on "
                         "unlabelled samples with f ~ U[5, 12]")
    ap.add_argument("--phase-lam", type=float, default=0.0,
                    help="supplementary exp. 3: weight of an OUT-OF-BAND phase "
                         "guidance penalty -- pushes the predicted correction "
                         "field's phase toward the analytic Born phase "
                         "|⟨u_hat,B1⟩|/(‖u_hat‖‖B1‖) on unlabelled samples "
                         "with f ~ U[5, 12]; magnitude is left free")
    ap.add_argument("--phase-order", type=int, default=1, choices=[1, 2],
                    help="Born order of the phase anchor: 1 = B1 only, "
                         "2 = B1 + B2 (the latter stays finite on the "
                         "convergent side of the Born series)");
    ap.add_argument("--p", type=int, default=128)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--n-train", type=int, default=None,
                    help="use a deterministic prefix of this many samples "
                         "(paired across bands); val = its trailing 10%%")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None,
                    help="checkpoint whose WEIGHTS initialise the model "
                         "(transfer-learning fine-tune, C4); the optimizer "
                         "and schedule start fresh")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = C.run_dir(a.name)
    if os.path.exists(os.path.join(out, "config.json")):
        print("run dir already initialised:", a.name)
    log = C.Logger(os.path.join(out, "train.log"))
    C.set_seed(a.seed)
    log("run=%s model=%s device=%s" % (a.name, a.model, dev))

    d = load_cache(a.data, dev)
    n = d["v"].shape[0]
    if a.n_train:
        n_val = max(64, int(a.n_train * a.val_frac))
        tr_idx = torch.arange(0, a.n_train, device=dev)
        val_idx = torch.arange(a.n_train, a.n_train + n_val, device=dev)
    else:
        g = torch.Generator().manual_seed(12345)
        perm = torch.randperm(n, generator=g)
        n_val = int(n * a.val_frac)
        val_idx, tr_idx = perm[:n_val].to(dev), perm[n_val:].to(dev)
    log("samples=%d train=%d val=%d  cache_max_res=%.2e"
        % (n, tr_idx.numel(), val_idx.numel(), d["max_res"]))

    kw = dict(in_ch=D.in_channels(a.f_emb, a.anchor), out_ch=2)
    if a.model == "fno":
        kw.update(width=a.width, modes=a.modes, n_layers=a.layers, proj=a.proj, norm=a.norm)
    elif a.model == "fno-film":
        kw.update(width=a.width, modes=a.modes, n_layers=a.layers, proj=a.proj, norm=a.norm)
    elif a.model == "deeponet":
        kw.update(width=a.width, p=a.p, n_grid=C.N)
    elif a.model == "unet":
        kw.update(base=a.base, depth=3)
    model = M.build_model(a.model, **kw).to(dev)
    if a.init:
        ck0 = torch.load(a.init, map_location=dev, weights_only=False)
        model.load_state_dict(ck0["model"])
        log("init weights from %s (source epoch %s, source val %s)"
            % (a.init, ck0.get("epoch"), ck0.get("val")))
    nparam = C.count_params(model)
    log("params=%.3fM  cfg=%s" % (nparam / 1e6, {k: v for k, v in kw.items()}))

    # Anchored arm: two complex coefficients [Re a, Im a, Re b, Im b] on the
    # analytic Born pair.  Initialised at (a, b) = (1, 0), i.e. the arm starts
    # at the plain first-order Born field and the network learns what it must
    # add.  Four numbers, trained on the band only -- they are the whole
    # zero-shot content of the anchor.
    ab = None
    if a.anchor == "born":
        ab = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev))
        log("anchor=born  a,b learnable  init a=(1,0) b=(0,0)")

    cfg = dict(**vars(a))
    cfg.update(model=a.model, params=int(nparam))
    cfg["n_train"] = int(tr_idx.numel())
    cfg["n_val"] = int(n_val)
    cfg["cache_max_res"] = d["max_res"]
    cfg["device"] = dev
    cfg["torch"] = torch.__version__
    C.save_json(cfg, os.path.join(out, "config.json"))

    opt = torch.optim.Adam((list(model.parameters()) + [ab] if ab is not None
                            else model.parameters()), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs,
                                                       eta_min=a.lr * 0.01)
    csv = open(os.path.join(out, "train_log.csv"), "w")
    csv.write("epoch,train_relL2,val_relL2,lr,sec\n")
    best = 1e9
    t0 = time.time()
    stop = False

    for ep in range(1, a.epochs + 1):
        model.train()
        gp = torch.randperm(tr_idx.numel(), generator=torch.Generator().manual_seed(
            a.seed * 100003 + ep))
        tot, cnt = 0.0, 0
        for s in range(0, tr_idx.numel(), a.batch):
            sel = tr_idx[gp[s:s + a.batch]]
            born = None
            if a.anchor == "none":
                x, psi_inc, psi_t = make_inputs(d, sel, emb=a.f_emb)
            else:
                x, psi_inc, psi_t, born = make_inputs(d, sel, emb=a.f_emb,
                                                      anchor=a.anchor)
            out_ = model(x)
            u = torch.complex(out_[:, 0], out_[:, 1])
            if a.anchor == "born":
                u = D.anchored_u(u, born, ab)
            psi_hat = psi_inc + u
            loss = rel_l2(psi_hat, psi_t).mean()
            if a.energy_lam > 0:
                # C3: calibrate the correction ENERGY on training samples.  The
                # relative-L2 loss alone is minimised long before the output
                # energy matches, and out of band the model extrapolates that
                # tolerance into a collapse toward zero correction (error ~ 1,
                # psi_hat ~ psi_inc).  A log-ratio penalty is symmetric in
                # over/under-prediction and scale-free.
                u_true = psi_t - psi_inc
                r = (torch.linalg.norm(u.reshape(sel.numel(), -1), dim=1) /
                     torch.linalg.norm(u_true.reshape(sel.numel(), -1),
                                       dim=1).clamp_min(1e-12))
                loss = loss + a.energy_lam * torch.mean(
                    torch.log(r.clamp_min(1e-12)) ** 2)
            if a.ob_energy_lam > 0:
                # C3b: the in-band penalty above is inactive once the model is
                # energy-calibrated in band (ratio ~ 1), so it cannot move the
                # out-of-band extrapolation.  This term draws UNLABELLED
                # out-of-band samples (f ~ U[5,12]) every step and anchors the
                # predicted correction energy on the first Born field, which is
                # computable without labels at any frequency.  Only the energy
                # is constrained -- the phase is left free -- so a recovery of
                # the energy ratio WITHOUT a recovery of the error isolates the
                # pessimistic-collapse hypothesis from phase information.
                # Parameters only: no LS solve, the arm never sees a label here.
                sd_ob = a.seed * 100003 + ep * 100 + s
                f_ob, vm_ob, th_ob = D.sample_params(16, 5.0, 12.0, 0.25, 0.25,
                                                     sd_ob, dev)
                g_ob = torch.Generator(device=dev).manual_seed(sd_ob + 7777)
                v_ob = P.make_contrast(C.N, 16, 1.0, C.MM_REF, g_ob,
                                       dev) * vm_ob[:, None, None]
                inc_ob = P.incident_plane_wave(C.N, 2 * np.pi * f_ob, th_ob,
                                               dev).to(torch.complex64)
                born = (P.born_approx(v_ob, 2 * np.pi * f_ob, th_ob, C.GAMMA,
                                      C.N, order=1, device=dev).detach()
                        - inc_ob)
                x_ob = D.build_inputs(v_ob.float(), inc_ob, f_ob)
                o_ob = model(x_ob)
                u_ob = torch.complex(o_ob[:, 0], o_ob[:, 1])
                nb = torch.linalg.norm(u_ob.reshape(16, -1), dim=1)
                nb0 = torch.linalg.norm(born.abs().reshape(16, -1), dim=1
                                        ).clamp_min(1e-12)
                loss = loss + a.ob_energy_lam * torch.mean(
                    torch.log((nb / nb0).clamp_min(1e-12)) ** 2)
            if a.phase_lam > 0:
                # supplementary exp. 3: phase guidance out of band.  C3b showed
                # the model lacks PHASE information, not energy, beyond the
                # band: forcing the correction energy toward the Born level
                # collapsed the alignment while leaving the error worse.  This
                # term constrains the phase directly -- |<u_hat, B1>|/(||u_hat||
                # ||B1||) toward 1 -- while leaving the magnitude free, on the
                # same label-free out-of-band samples.  If phase is the missing
                # ingredient, the arm should recover alignment without paying
                # the energy mismatch that broke the C3b arm.
                sd_p = a.seed * 100003 + ep * 100 + s + 31
                f_ob, vm_ob, th_ob = D.sample_params(16, 5.0, 12.0, 0.25, 0.25,
                                                     sd_p, dev)
                g_ob = torch.Generator(device=dev).manual_seed(sd_p + 7777)
                v_ob = P.make_contrast(C.N, 16, 1.0, C.MM_REF, g_ob,
                                       dev) * vm_ob[:, None, None]
                inc_ob = P.incident_plane_wave(C.N, 2 * np.pi * f_ob, th_ob,
                                               dev).to(torch.complex64)
                b1 = (P.born_approx(v_ob, 2 * np.pi * f_ob, th_ob, C.GAMMA,
                                    C.N, order=a.phase_order, device=dev).detach()
                      - inc_ob)
                x_ob = D.build_inputs(v_ob.float(), inc_ob, f_ob)
                o_ob = model(x_ob)
                u_ob = torch.complex(o_ob[:, 0], o_ob[:, 1])
                num = torch.abs(torch.sum(torch.conj(u_ob) * b1, dim=(-2, -1)))
                den = (torch.linalg.norm(u_ob.reshape(16, -1), dim=1).clamp_min(
                    1e-12) * torch.linalg.norm(b1.reshape(16, -1), dim=1
                    ).clamp_min(1e-12))
                loss = loss + a.phase_lam * torch.mean(1.0 - num / den)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (list(model.parameters()) + [ab] if ab is not None
                 else model.parameters()), 1.0)
            opt.step()
            tot += float(loss) * sel.numel()
            cnt += sel.numel()
        sched.step()
        tr = tot / max(cnt, 1)

        model.eval()
        with torch.no_grad():
            vals = []
            for s in range(0, val_idx.numel(), 64):
                sel = val_idx[s:s + 64]
                born = None
                if a.anchor == "none":
                    x, psi_inc, psi_t = make_inputs(d, sel, emb=a.f_emb)
                else:
                    x, psi_inc, psi_t, born = make_inputs(d, sel, emb=a.f_emb,
                                                          anchor=a.anchor)
                o = model(x)
                u = torch.complex(o[:, 0], o[:, 1])
                if a.anchor == "born":
                    u = D.anchored_u(u, born, ab)
                vals.append(rel_l2(psi_inc + u, psi_t))
            va = float(torch.cat(vals).mean())
        csv.write("%d,%.6e,%.6e,%.6e,%.1f\n" % (ep, tr, va, sched.get_last_lr()[0],
                                                time.time() - t0))
        csv.flush()
        if va < best:
            best = va
            ck = dict(model=model.state_dict(), cfg=cfg, epoch=ep,
                      val=va, params=nparam)
            if ab is not None:
                # the anchor's two complex coefficients travel with the weights:
                # they are part of the predictor, and evaluation must apply the
                # same analytic term it was trained with
                ck["anchor_ab"] = ab.detach().cpu()
            torch.save(ck, os.path.join(out, "checkpoint.pt"))
        if ep % 10 == 0 or ep == 1:
            log("ep %3d  train=%.4e  val=%.4e  best=%.4e  %.0fs"
                % (ep, tr, va, best, time.time() - t0))
        if a.budget is not None and time.time() - t0 > a.budget:
            log("budget reached at epoch %d" % ep)
            stop = True
            break

    csv.close()
    res = dict(name=a.name, best_val=best, epochs_done=ep, params=int(nparam),
               sec=time.time() - t0, stopped_early=stop)
    C.save_json(res, os.path.join(out, "result.json"))
    log("DONE best_val=%.4e  (%.0fs)" % (best, time.time() - t0))


if __name__ == "__main__":
    main()
