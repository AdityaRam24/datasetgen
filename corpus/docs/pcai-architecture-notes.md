# HPE Private Cloud AI — Architecture Notes

These notes exist so that a fresh clone of this project has something to build
from. Replace them with your real documentation; drop PDFs straight into this
folder and they will be parsed on the next run.

## Control plane and data plane

PCAI separates a control plane (cluster lifecycle, identity, tenancy) from the
AI service plane (MLIS for inference, MLDM for data pipelines, MLDE for
distributed training). The control plane runs on the management nodes; the
service plane is scheduled across the GPU worker pool.

## Identity and access

Authentication is brokered through Keycloak. Each AI service trusts Keycloak as
its OIDC provider rather than holding its own user database. Role bindings are
mapped from Keycloak groups to Kubernetes RBAC at namespace granularity, so a
team that can deploy an MLIS endpoint cannot necessarily read another team's
lakehouse bucket.

## Storage

The lakehouse presents an S3-compatible object interface. Model artifacts,
datasets and pipeline outputs all land there. Endpoints receive scoped
credentials as Kubernetes secrets; those credentials expire, which is the most
common reason a previously working endpoint starts failing to start.

## Networking and ingress

External traffic terminates at the ingress controller, which holds the platform
TLS certificate. Endpoints are exposed as subpaths or subdomains depending on
the install profile. In an air-gapped install, no component may reach the public
internet, so container images and model artifacts must be pre-staged into the
internal registry and the lakehouse before deployment.

## Sizing

GPU worker sizing is driven by concurrent inference endpoints rather than by
total model count: an idle endpoint still holds its GPU allocation until it is
scaled to zero. Plan for headroom equal to at least one full node so that a
rolling upgrade can drain a node without evicting production endpoints.

## Upgrades and backup

Upgrades are staged control plane first, then service plane. Back up the
Keycloak realm and the platform configuration before an upgrade; model
artifacts in the lakehouse are not part of the platform backup and are assumed
to be reproducible from their source pipelines.
