# Content Delivery Networks

A content delivery network, or CDN, serves cached files from locations near users. It is helpful when a web application distributes the same images, documents, JavaScript bundles, or video files to many visitors.

The origin server remains the authoritative source. When a requested object is not in a nearby cache, the CDN fetches it from the origin and may keep a copy for the cache lifetime. Cache-control headers determine how long browsers and edge locations can reuse a response.

CDNs do not replace access control. Private files should use short-lived signed URLs or an authenticated proxy. Public portfolio assets can usually use long cache lifetimes, while frequently changing content needs cache invalidation or versioned filenames.

Cache keys need careful design. Query parameters, headers, cookies, and content negotiation can affect whether two requests are equivalent. Accidentally omitting a parameter can serve the wrong representation, while including unnecessary headers can reduce the cache hit rate. Compression should be configured so a response is not served in a format that a client cannot decode.

A CDN is a delivery layer, not a source of truth. Private files should use short-lived signed URLs, and the origin should not be publicly reachable when that can be avoided. Useful operational measurements include cache hit ratio, origin latency, bandwidth, purge activity, and error rate. These measurements show whether the CDN is actually reducing load rather than merely adding another component.