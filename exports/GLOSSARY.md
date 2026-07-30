# Glossary — Training/RAG dataset for the Kalam PCAI assistant

7 terms, generated 2026-07-30T20:19:56Z by the local model `qwen2.5-coder:7b`. Every definition is drawn from the sources cited beneath it — nothing here is the model's own knowledge.

[A](#a)  [G](#g)  [I](#i)  [M](#m)  [S](#s)  [T](#t)

## A

### air-gapped install

An installation where no component may reach the public internet, requiring pre-staging of container images and model artifacts into internal registries and the lakehouse before deployment.

<sub>Sources: [pcai architecture notes](file:///C:/Users/Steve/Desktop/kalam/dataset-generation/corpus/docs/pcai-architecture-notes.md)</sub>

## G

### GPU worker sizing

The process of determining the number of GPU workers needed based on concurrent inference endpoints rather than total model count.

<sub>Sources: [pcai architecture notes](file:///C:/Users/Steve/Desktop/kalam/dataset-generation/corpus/docs/pcai-architecture-notes.md)</sub>

## I

### ingress controller

A service that terminates external traffic to the platform and holds the platform TLS certificate.

<sub>Sources: [pcai architecture notes](file:///C:/Users/Steve/Desktop/kalam/dataset-generation/corpus/docs/pcai-architecture-notes.md)</sub>

## M

### MLDE

MLDE is a service that facilitates distributed training of machine learning models across GPU worker nodes.

*Also known as: distributed training*

<sub>Sources: [pcai architecture notes](file:///C:/Users/Steve/Desktop/kalam/dataset-generation/corpus/docs/pcai-architecture-notes.md)</sub>

### MLDM

MLDM is a service that manages data pipelines for machine learning workflows.

*Also known as: data pipelines*

<sub>Sources: [pcai architecture notes](file:///C:/Users/Steve/Desktop/kalam/dataset-generation/corpus/docs/pcai-architecture-notes.md)</sub>

## S

### statement

In PgBouncer, statement pooling mode forbids multi-statement transactions entirely.

<sub>Sources: [PgBouncer pooling modes](input://note/pgbouncer-pooling-modes-c2109cb2)</sub>

## T

### transaction

In PgBouncer, transaction pooling mode returns the server connection to the pool at commit, allowing a small pool to serve many clients.

<sub>Sources: [PgBouncer pooling modes](input://note/pgbouncer-pooling-modes-c2109cb2)</sub>
