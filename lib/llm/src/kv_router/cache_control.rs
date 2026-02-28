// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::{
    component::Component,
    pipeline::{PushRouter, RouterMode, SingleIn},
    protocols::annotated::Annotated,
};
use futures::StreamExt;

use crate::protocols::TokenIdType;

/// State captured at routing time for a deferred PIN after generation completes.
pub(crate) struct PinState {
    pub token_ids: Vec<TokenIdType>,
    pub cc_client: CacheControlClient,
    pub instance_id: u64,
    pub ttl_seconds: u64,
}

/// State captured at routing time for a deferred cache lifecycle action after generation.
pub(crate) struct CacheActionState {
    pub action: String,
    pub token_ids: Vec<TokenIdType>,
    pub cc_client: CacheControlClient,
    pub instance_id: u64,
    pub prefix_id: Option<String>,
}

/// A PushRouter client typed for cache_control requests/responses.
///
/// Both request and response are untyped JSON. The worker's cache_control
/// endpoint returns {"status": "ok"/"error", ...} but the router treats
/// PIN as fire-and-forget and only logs the response at debug level.
pub type CacheControlClient = PushRouter<serde_json::Value, Annotated<serde_json::Value>>;

/// Create a cache_control client from a component.
///
/// Connects to the "cache_control" endpoint on the given component and returns
/// a PushRouter client for sending cache control operations (pin_prefix,
/// unpin_prefix) to workers.
pub(crate) async fn create_cache_control_client(
    component: &Component,
) -> Result<CacheControlClient> {
    let client = component.endpoint("cache_control").client().await?;
    CacheControlClient::from_client(client, RouterMode::KV).await
}

/// Fire-and-forget pin_prefix to the worker that served this request.
///
/// Spawns a detached task that sends the pin request and logs the outcome.
/// Does nothing if `client` is `None` (logs a warning).
pub fn spawn_pin_prefix(
    client: Option<&CacheControlClient>,
    token_ids: &[TokenIdType],
    instance_id: u64,
    context_id: &str,
    ttl_seconds: u64,
) {
    let Some(cc) = client else {
        tracing::warn!(
            request_id = %context_id,
            "cache_control set but no cache_control_client configured"
        );
        return;
    };

    let cc = cc.clone();
    let token_ids = token_ids.to_vec();
    let context_id = context_id.to_owned();

    tokio::spawn(async move {
        let pin_request = serde_json::json!({
            "action": "pin_prefix",
            "token_ids": token_ids,
            "ttl_seconds": ttl_seconds,
        });
        match cc.direct(SingleIn::new(pin_request), instance_id).await {
            Ok(mut stream) => {
                if let Some(resp) = stream.next().await {
                    tracing::info!(
                        request_id = %context_id,
                        worker_id = instance_id,
                        ?resp,
                        "pin_prefix response"
                    );
                }
                // Drain remaining stream to avoid "Failed to publish
                // complete final" errors from the push handler.
                while stream.next().await.is_some() {}
            }
            Err(e) => {
                tracing::warn!(
                    request_id = %context_id,
                    worker_id = instance_id,
                    "Failed to pin prefix: {e}"
                );
            }
        }
    });
}

/// Fire-and-forget cache lifecycle action to a specific worker.
///
/// Maps `cache_action` values from `agent_hints` to worker-side `cache_control` actions:
/// - "demote_to_host"    -> action: "demote_prefix", target: "host"
/// - "demote_to_storage" -> action: "demote_prefix", target: "storage"
/// - "evict"             -> action: "evict_prefix"
/// - "promote"           -> action: "promote_prefix"
///
/// Unknown actions are logged and silently dropped.
pub fn spawn_cache_action(
    client: &CacheControlClient,
    cache_action: &str,
    token_ids: &[TokenIdType],
    instance_id: u64,
    context_id: &str,
    prefix_id: Option<&str>,
) {
    let (action, target) = match cache_action {
        "demote_to_host" => ("demote_prefix", Some("host")),
        "demote_to_storage" => ("demote_prefix", Some("storage")),
        "evict" => ("evict_prefix", None),
        "promote" => ("promote_prefix", None),
        other => {
            tracing::warn!(
                request_id = %context_id,
                cache_action = other,
                "Unknown cache_action, ignoring"
            );
            return;
        }
    };

    let cc = client.clone();
    let token_ids = token_ids.to_vec();
    let context_id = context_id.to_owned();
    let action = action.to_string();
    let target = target.map(|s| s.to_string());
    let prefix_id = prefix_id.map(|s| s.to_owned());

    tokio::spawn(async move {
        let mut payload = serde_json::json!({
            "action": action,
            "token_ids": token_ids,
        });
        if let Some(ref target) = target {
            payload
                .as_object_mut()
                .unwrap()
                .insert("target".to_string(), serde_json::json!(target));
        }
        if let Some(ref prefix_id) = prefix_id {
            payload
                .as_object_mut()
                .unwrap()
                .insert("prefix_id".to_string(), serde_json::json!(prefix_id));
        }

        tracing::debug!(
            request_id = %context_id,
            %action,
            instance_id,
            "Sending cache_action"
        );

        match cc.direct(SingleIn::new(payload), instance_id).await {
            Ok(mut stream) => {
                if let Some(resp) = stream.next().await {
                    tracing::info!(
                        request_id = %context_id,
                        %action,
                        worker_id = instance_id,
                        ?resp,
                        "cache_action response"
                    );
                }
                // Drain remaining stream to avoid push handler errors.
                while stream.next().await.is_some() {}
            }
            Err(e) => {
                tracing::warn!(
                    request_id = %context_id,
                    %action,
                    worker_id = instance_id,
                    "cache_action dispatch failed: {e}"
                );
            }
        }
    });
}

/// Fire-and-forget register_owner to stamp prefix ownership on cached tree nodes.
///
/// Sends a `register_owner` action to the worker so that the scheduler can
/// track which agents own which radix tree nodes. This enables prefix_id-aware
/// eviction where shared nodes (owned by multiple agents) are preserved.
pub fn spawn_register_owner(
    client: &CacheControlClient,
    token_ids: &[TokenIdType],
    instance_id: u64,
    prefix_id: &str,
    context_id: &str,
) {
    let cc = client.clone();
    let token_ids = token_ids.to_vec();
    let context_id = context_id.to_owned();
    let prefix_id = prefix_id.to_owned();

    tokio::spawn(async move {
        let payload = serde_json::json!({
            "action": "register_owner",
            "token_ids": token_ids,
            "prefix_id": prefix_id,
        });

        tracing::debug!(
            request_id = %context_id,
            %prefix_id,
            instance_id,
            "Sending register_owner"
        );

        match cc.direct(SingleIn::new(payload), instance_id).await {
            Ok(mut stream) => {
                // Drain stream to avoid push handler errors.
                while stream.next().await.is_some() {}
            }
            Err(e) => {
                tracing::warn!(
                    request_id = %context_id,
                    %prefix_id,
                    worker_id = instance_id,
                    "register_owner dispatch failed: {e}"
                );
            }
        }
    });
}
