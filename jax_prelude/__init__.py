import jax
import jax.numpy as jnp

smp = jax.devices("cpu")[0]
gpus = jax.devices("cuda")
gpu = gpus[0]

bf16 = jnp.dtype("bfloat16")
f16 = jnp.dtype("float16")
f32 = jnp.dtype("float32")
f64 = jnp.dtype("float64")
i8 = jnp.dtype("int8")
i16 = jnp.dtype("int16")
i32 = jnp.dtype("int32")
i64 = jnp.dtype("int64")
u8 = jnp.dtype("uint8")
u16 = jnp.dtype("uint16")
u32 = jnp.dtype("uint32")
u64 = jnp.dtype("uint64")
