import os
import sys
import numpy as np
import cv2
import mujoco

# Adiciona o diretório raiz ao path para importar as funções do projeto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ur3_push_mujoco.utils import mujoco_utils

def main():
    # 1. Gerar o XML estático do ambiente
    xml_str = mujoco_utils.generate_model_xml_string(
        use_sim_config=True,
        include_camera=True,
        obj_type="cylinder",
        obj_xy_pos=np.array([0.45, 0.1]),   # Coloca o cilindro dentro da área
        target_xy_pos=np.array([0.3, -0.15]) # Coloca o target também dentro
    )
    
    # Injetar aumento de framebuffer offscreen para suportar 1920x1080
    xml_str = xml_str.replace('</mujoco>', '<visual><global offwidth="1920" offheight="1080"/></visual></mujoco>')
    
    
    # Mudar diretório de trabalho temporariamente para o MuJoCo achar os arquivos .STL
    import ur3_push_mujoco
    assets_dir = os.path.join(ur3_push_mujoco.__path__[0], "assets")
    original_cwd = os.getcwd()
    os.chdir(assets_dir)
    
    # 2. Carregar modelo e dados na CPU
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    os.chdir(original_cwd)
    
    # 2.5. Configurar o robô para a pose inicial do treinamento
    _INITIAL_QPOS = {
        "shoulder_pan_joint":        0.0,
        "shoulder_lift_joint":      -1.5708,
        "elbow_joint":               1.5708,
        "wrist_1_joint":            -1.5708,
        "wrist_2_joint":            -1.5708,
        "wrist_3_joint":             0.0,
    }
    for name, val in _INITIAL_QPOS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            qadr = model.jnt_qposadr[joint_id]
            data.qpos[qadr] = val
    
    # 3. Avançar a cinemática para aplicar a posição inicial
    mujoco.mj_forward(model, data)
    
    # 4. Inicializar o renderizador off-screen do MuJoCo
    renderer = mujoco.Renderer(model, height=1080, width=1920)
    
    # Configurar uma Câmera Livre (Free Camera) para Visão Top-Down (Perpendicular)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = np.array([0.35, 0.0, 0.0]) # Olhando para o centro da área de spawn
    cam.distance = 1.2 # Zoom out (Subindo a câmera no eixo Z)
    cam.azimuth = 0 # Rotacionado em 90 graus no eixo perpendicular
    cam.elevation = -90 # Perpendicular de cima para baixo
    
    # Atualiza a cena usando a câmera livre customizada
    renderer.update_scene(data, camera=cam)
    
    # 5. Injetar a zona de Spawn (Caixa Azul Transparente) diretamente na Cena Renderizada
    scene = renderer.scene
    if scene.ngeom < scene.maxgeom:
        # Dimensões da área (X: 0.25 a 0.50 | Y: -0.20 a 0.20)
        # O centro de X é (0.25+0.50)/2 = 0.375
        # O centro de Y é 0.0
        # A altura Z é a da mesa (0.02) + um leve offset para não piscar (0.001)
        pos = np.array([0.375, 0.0, 0.021])
        
        # O tamanho no MuJoCo é a "metade" da dimensão (half-extents)
        # Largura X = 0.50 - 0.25 = 0.25 -> Metade = 0.125
        # Comprimento Y = 0.20 - (-0.20) = 0.40 -> Metade = 0.20
        # Espessura Z = 0.001
        size = np.array([0.125, 0.20, 0.001])
        
        # Cor: Azul claro, 40% de opacidade (RGBA)
        rgba = np.array([0.2, 0.6, 1.0, 0.4])
        
        # Matriz de rotação identidade (alinhada com a mesa)
        mat = np.eye(3).flatten()
        
        # Adicionar à cena
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_BOX,
            size,
            pos,
            mat,
            rgba
        )
        scene.ngeom += 1

    # 6. Renderizar os pixels
    pixels = renderer.render()
    
    # 7. Salvar usando OpenCV (O MuJoCo devolve RGB, o OpenCV salva em BGR)
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'area_de_spawn.png'))
    cv2.imwrite(output_path, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    
    print(f"Sucesso! Imagem gerada em: {output_path}")

if __name__ == '__main__':
    main()
