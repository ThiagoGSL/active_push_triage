import mujoco
import mujoco_warp as mjw
import warp as wp

wp.init()

xml = """
<mujoco>
  <worldbody>
    <geom type="box" size="1 1 1" mass="1.0" friction="1 0.5 0.5"/>
  </worldbody>
</mujoco>
"""
mjm = mujoco.MjModel.from_xml_string(xml)

# Create batched model on GPU
m_gpu = mjw.put_model(mjm)

print("--- ANALISANDO PROPRIEDADES FÍSICAS (MASSA) ---")
print(f"Shape de body_mass: {m_gpu.body_mass.shape}")
print(f"Strides de body_mass: {m_gpu.body_mass.strides}")

print("\n--- ANALISANDO PROPRIEDADES FÍSICAS (ATRITO) ---")
print(f"Shape de geom_friction: {m_gpu.geom_friction.shape}")
print(f"Strides de geom_friction: {m_gpu.geom_friction.strides}")

print("\nConclusão de Memória:")
if m_gpu.body_mass.strides[0] == 0:
    print("-> A dimensão de batch (ambientes) tem stride 0.")
    print("-> Isso significa que TODOS os mundos apontam para o mesmo exato endereço de memória na GPU.")
    print("-> Logo, não é possível ter massas diferentes para ambientes paralelos.")
