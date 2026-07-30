# MLIS Inference Endpoint Returns 503

## Symptom

A deployed MLIS endpoint returns `503 Service Unavailable` for every request.
`kubectl get pods -n mlis` shows the serving pod in `CrashLoopBackOff`, or
`Running` but never `1/1 Ready`. Callers see:

    upstream connect error or disconnect/reset before headers. reset reason: connection failure

## Likely cause

In order of how often it is actually the problem:

1. The model artifact could not be pulled from the S3-compatible lakehouse
   bucket — expired or wrong credentials in the endpoint's secret.
2. The GPU the endpoint requested is already fully allocated, so the pod is
   `Pending` and the service has no healthy backend.
3. The container image is larger than the node's ephemeral storage and the
   kubelet evicted it.
4. The readiness probe timeout is shorter than the model load time, so a large
   model never becomes ready even though nothing is wrong.

## Preconditions

- `kubectl` context pointing at the PCAI cluster
- Access to the `mlis` namespace
- The endpoint name, referred to below as `$ENDPOINT`

## Resolution steps

1. Identify the failing pod:
   `kubectl get pods -n mlis -l app=$ENDPOINT -o wide`
2. Read the last termination reason:
   `kubectl describe pod -n mlis <pod> | sed -n '/Last State/,/Events/p'`
3. Check the init container that pulls the artifact:
   `kubectl logs -n mlis <pod> -c model-fetch --tail=100`
4. If the logs show `AccessDenied` or `InvalidAccessKeyId`, refresh the bucket
   credentials: `kubectl -n mlis delete secret $ENDPOINT-s3` and re-create it
   from the MLIS UI under Endpoint > Storage > Reconnect.
5. If the pod is `Pending`, confirm GPU availability:
   `kubectl describe node | grep -A3 "nvidia.com/gpu"`
   Free capacity by scaling an idle endpoint to zero replicas:
   `kubectl -n mlis scale deploy <idle-endpoint> --replicas=0`
6. If the kubelet evicted the pod for disk pressure, prune images on the node:
   `crictl rmi --prune`
7. If the model loads but never passes readiness, raise the probe budget:
   `kubectl -n mlis patch deploy $ENDPOINT --type=json -p='[{"op":"replace","path":"/spec/template/spec/containers/0/readinessProbe/initialDelaySeconds","value":300}]'`

## Verification

- `kubectl get pods -n mlis -l app=$ENDPOINT` shows `1/1 Running`
- `curl -s -o /dev/null -w '%{http_code}' https://$ENDPOINT/v1/models` returns `200`
- A test inference returns a payload within the endpoint's latency SLO

## Rollback

If the patched probe budget makes deployments hang, revert it:

    kubectl -n mlis rollout undo deploy $ENDPOINT

Rolling back does not delete the endpoint or its model artifact.

## Escalation

If the artifact fetch keeps failing after credentials are refreshed, the
lakehouse bucket policy is the likely culprit — escalate to the platform team
that owns the object store, not to the MLIS on-call.
