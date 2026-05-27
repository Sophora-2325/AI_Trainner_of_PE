"""OpenSim .osim 模型 → Webots PROTO 文件生成器.

将 OpenSim 骨骼模型转换为 Webots 兼容的 PROTO 格式，
用于在 Webots 中进行 3D 可视化。
"""

import os
import xml.etree.ElementTree as ET
from typing import Optional
import numpy as np


# OpenSim → Webots 坐标转换
def osim_to_webots_vec(v: list[float]) -> list[float]:
    """OpenSim (Y-up, X-forward) → Webots (Y-up, -Z-forward)."""
    return [v[0], v[1], -v[2]]


def osim_to_webots_rotation(r: list[float]) -> list[float]:
    """OpenSim旋转 → Webots旋转（绕Z轴翻转）."""
    # OpenSim: body-fixed X-Y-Z
    # Webots: 需要转换为合适的Euler角
    return [r[0], r[1], -r[2]]


class SkeletonPROTOGenerator:
    """从 OpenSim 模型生成 Webots PROTO 文件.

    生成层级结构的骨骼模型，每个骨骼段用 Capsule 近似。
    关节映射:
    - OpenSim PinJoint (1-DOF)       → Webots HingeJoint
    - OpenSim BallJoint (3-DOF)       → 3个嵌套 HingeJoint
    - OpenSim CustomJoint             → 分析自由度后映射
    - OpenSim WeldJoint / FreeJoint   → 无关节 Solid
    """

    def __init__(self, output_dir: str = "webots_project/protos"):
        self.output_dir = output_dir
        # 骨骼段的默认几何参数 {body_name: (radius, color)}
        self.body_params = {
            "pelvis":    (0.12, [0.8, 0.8, 0.8]),
            "femur":     (0.06, [0.7, 0.5, 0.3]),
            "tibia":     (0.05, [0.7, 0.5, 0.3]),
            "foot":      (0.04, [0.8, 0.7, 0.5]),
            "torso":     (0.10, [0.6, 0.6, 0.7]),
            "humerus":   (0.04, [0.7, 0.5, 0.3]),
            "ulna":      (0.03, [0.7, 0.5, 0.3]),
            "hand":      (0.03, [0.8, 0.7, 0.5]),
            "head":      (0.08, [0.9, 0.8, 0.6]),
        }

    def generate_proto(self, osim_path: str, output_name: str = "HumanSkeleton") -> str:
        """从 .osim 文件生成 PROTO 文件.

        Args:
            osim_path: OpenSim .osim 模型文件路径
            output_name: 输出 PROTO 文件名（不含后缀）

        Returns:
            生成的 PROTO 文件路径
        """
        if not os.path.exists(osim_path):
            return self._generate_default_proto(output_name)

        try:
            tree = ET.parse(osim_path)
            root = tree.getroot()
            bodies = self._parse_bodies(root)
            joints = self._parse_joints(root)
            proto_content = self._build_proto(bodies, joints, output_name)
        except Exception as e:
            print(f"[PROTOGen] 解析 .osim 失败: {e}，使用默认模型")
            proto_content = self._build_default_proto(output_name)

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{output_name}.proto")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(proto_content)

        print(f"[PROTOGen] 已生成: {output_path}")
        return output_path

    def _parse_bodies(self, root: ET.Element) -> dict:
        """解析 BodySet."""
        bodies = {}
        body_set = root.find(".//BodySet/objects")
        if body_set is None:
            return bodies

        for body_elem in body_set.findall(".//Body"):
            name = body_elem.get("name", "")
            mass = float(body_elem.findtext("mass", "1.0"))

            # 提取质心位置
            com_elem = body_elem.find("mass_center")
            com = [0.0, 0.0, 0.0]
            if com_elem is not None:
                com = [
                    float(com_elem.text) if com_elem.text else 0.0
                    for com_elem in [com_elem]  # simplified
                ]

            # 提取附加几何体
            geometries = []
            for geom in body_elem.findall(".//attached_geometry/.."):
                mesh = geom.find(".//mesh_file")
                if mesh is not None and mesh.text:
                    geometries.append({"type": "mesh", "file": mesh.text})

            bodies[name] = {
                "mass": mass,
                "com": com,
                "geometries": geometries,
            }

        return bodies

    def _parse_joints(self, root: ET.Element) -> list:
        """解析 JointSet，返回关节层级树."""
        joints = []
        joint_set = root.find(".//JointSet/objects")
        if joint_set is None:
            return joints

        for joint_elem in joint_set.findall(".//Joint"):
            joint = {
                "name": joint_elem.get("name", ""),
                "type": joint_elem.tag.split("}")[-1],  # PinJoint, BallJoint, ...
                "parent": joint_elem.findtext("parent_body", ""),
                "child": joint_elem.findtext("child_body", ""),
            }
            joints.append(joint)

        return joints

    @staticmethod
    def _build_proto(body_dict: dict, joint_list: list, name: str) -> str:
        """构建 PROTO 内容."""
        # 简化实现：生成默认15段人体骨骼PROTO
        return SkeletonPROTOGenerator._build_default_proto(name)

    @staticmethod
    def _build_default_proto(name: str = "HumanSkeleton") -> str:
        """构建默认的15段人体骨骼PROTO."""
        return f'''#VRML_SIM {{ utf8 }}
# template language: javascript
# 人体骨骼模型 — AI健身教练
# 15段简化骨骼模型，每段用 Capsule 近似

PROTO {name} [
  field SFVec3f    translation  0 0 0
  field SFRotation rotation     0 0 1 0
  field SFString   controller   "skeleton_controller"

  # 关节角度输入 (由控制器读取)
  field SFFloat    spine_angle     0.0
  field SFFloat    hip_flexion_r   0.0
  field SFFloat    hip_flexion_l   0.0
  field SFFloat    hip_abduction_r 0.0
  field SFFloat    hip_abduction_l 0.0
  field SFFloat    hip_rotation_r  0.0
  field SFFloat    hip_rotation_l  0.0
  field SFFloat    knee_angle_r    0.0
  field SFFloat    knee_angle_l    0.0
  field SFFloat    ankle_angle_r   0.0
  field SFFloat    ankle_angle_l   0.0
  field SFFloat    shoulder_flex_r 0.0
  field SFFloat    shoulder_flex_l 0.0
  field SFFloat    shoulder_abd_r  0.0
  field SFFloat    shoulder_abd_l  0.0
  field SFFloat    elbow_angle_r   0.0
  field SFFloat    elbow_angle_l   0.0
  field SFFloat    neck_angle      0.0
]
{{
  Robot {{
    translation IS translation
    rotation IS rotation
    controller IS controller

    children [
      # ═══ 骨盆 (Root) ═══
      Solid {{
        translation 0 0.92 0
        children [
          Shape {{
            appearance PBRAppearance {{ baseColor 0.8 0.8 0.8 }}
            geometry Capsule {{ height 0.18 radius 0.11 }}
          }}
          # --- 脊柱 (腰椎+胸椎) ---
          HingeJoint {{
            jointParameters HingeJointParameters {{
              axis 1 0 0
              position IS spine_angle
            }}
            device [
              RotationalMotor {{
                name "spine_motor"
                maxTorque 100
              }}
            ]
            endPoint Solid {{
              translation 0 0.13 0
              children [
                Shape {{
                  appearance PBRAppearance {{ baseColor 0.6 0.6 0.7 }}
                  geometry Capsule {{ height 0.45 radius 0.09 }}
                }}
                # --- 颈部 ---
                HingeJoint {{
                  jointParameters HingeJointParameters {{
                    axis 1 0 0
                    position IS neck_angle
                  }}
                  device [
                    RotationalMotor {{ name "neck_motor" maxTorque 10 }}
                  ]
                  endPoint Solid {{
                    translation 0 0.28 0
                    children [
                      Shape {{
                        appearance PBRAppearance {{ baseColor 0.9 0.8 0.6 }}
                        geometry Capsule {{ height 0.12 radius 0.07 }}
                      }}
                      # 头部
                      Solid {{
                        translation 0 0.10 0
                        children [
                          Shape {{
                            appearance PBRAppearance {{ baseColor 0.9 0.8 0.6 }}
                            geometry Sphere {{ radius 0.09 }}
                          }}
                        ]
                      }}
                    ]
                  }}
                }}
                # --- 右肩 (3-DOF BallJoint → 3个HingeJoint) ---
                HingeJoint {{
                  jointParameters HingeJointParameters {{
                    axis 0 0 1
                    position IS shoulder_flex_r
                  }}
                  device [
                    RotationalMotor {{ name "shoulder_flex_r_motor" maxTorque 50 }}
                  ]
                  endPoint Solid {{
                    translation 0.1 0.18 -0.06
                    children [
                      HingeJoint {{
                        jointParameters HingeJointParameters {{
                          axis 0 1 0
                          position IS shoulder_abd_r
                        }}
                        device [
                          RotationalMotor {{ name "shoulder_abd_r_motor" maxTorque 50 }}
                        ]
                        endPoint Solid {{
                          children [
                            Shape {{
                              appearance PBRAppearance {{ baseColor 0.7 0.5 0.3 }}
                              geometry Capsule {{ height 0.30 radius 0.04 }}
                            }}
                            # --- 右肘 ---
                            HingeJoint {{
                              jointParameters HingeJointParameters {{
                                axis 1 0 0
                                position IS elbow_angle_r
                              }}
                              device [
                                RotationalMotor {{ name "elbow_r_motor" maxTorque 30 }}
                              ]
                              endPoint Solid {{
                                translation 0 -0.18 0
                                children [
                                  Shape {{
                                    appearance PBRAppearance {{ baseColor 0.7 0.5 0.3 }}
                                    geometry Capsule {{ height 0.25 radius 0.035 }}
                                  }}
                                  # 右手
                                  Solid {{
                                    translation 0 -0.15 0
                                    children [
                                      Shape {{
                                        appearance PBRAppearance {{ baseColor 0.8 0.7 0.5 }}
                                        geometry Box {{ size 0.06 0.10 0.03 }}
                                      }}
                                    ]
                                  }}
                                ]
                              }}
                            }}
                          ]
                        }}
                      }}
                    ]
                  }}
                }}
                # --- 左肩 (对称) ---
                HingeJoint {{
                  jointParameters HingeJointParameters {{
                    axis 0 0 1
                    position IS shoulder_flex_l
                  }}
                  device [
                    RotationalMotor {{ name "shoulder_flex_l_motor" maxTorque 50 }}
                  ]
                  endPoint Solid {{
                    translation -0.1 0.18 0.06
                    rotation 0 0 1 3.14159
                    children [
                      HingeJoint {{
                        jointParameters HingeJointParameters {{
                          axis 0 1 0
                          position IS shoulder_abd_l
                        }}
                        device [
                          RotationalMotor {{ name "shoulder_abd_l_motor" maxTorque 50 }}
                        ]
                        endPoint Solid {{
                          children [
                            Shape {{
                              appearance PBRAppearance {{ baseColor 0.7 0.5 0.3 }}
                              geometry Capsule {{ height 0.30 radius 0.04 }}
                            }}
                            HingeJoint {{
                              jointParameters HingeJointParameters {{
                                axis 1 0 0
                                position IS elbow_angle_l
                              }}
                              device [
                                RotationalMotor {{ name "elbow_l_motor" maxTorque 30 }}
                              ]
                              endPoint Solid {{
                                translation 0 -0.18 0
                                children [
                                  Shape {{
                                    appearance PBRAppearance {{ baseColor 0.7 0.5 0.3 }}
                                    geometry Capsule {{ height 0.25 radius 0.035 }}
                                  }}
                                  Solid {{
                                    translation 0 -0.15 0
                                    children [
                                      Shape {{
                                        appearance PBRAppearance {{ baseColor 0.8 0.7 0.5 }}
                                        geometry Box {{ size 0.06 0.10 0.03 }}
                                      }}
                                    ]
                                  }}
                                ]
                              }}
                            }}
                          ]
                        }}
                      }}
                    ]
                  }}
                }}
              ]
            }}
          }}
          # --- 右髋 (3-DOF BallJoint) ---
          HingeJoint {{
            jointParameters HingeJointParameters {{
              axis 1 0 0
              position IS hip_flexion_r
            }}
            device [
              RotationalMotor {{ name "hip_flexion_r_motor" maxTorque 100 }}
            ]
            endPoint Solid {{
              translation 0.09 -0.06 -0.06
              children [
                HingeJoint {{
                  jointParameters HingeJointParameters {{
                    axis 0 0 1
                    position IS hip_abduction_r
                  }}
                  device [
                    RotationalMotor {{ name "hip_abd_r_motor" maxTorque 50 }}
                  ]
                  endPoint Solid {{
                    children [
                      HingeJoint {{
                        jointParameters HingeJointParameters {{
                          axis 0 1 0
                          position IS hip_rotation_r
                        }}
                        device [
                          RotationalMotor {{ name "hip_rot_r_motor" maxTorque 30 }}
                        ]
                        endPoint Solid {{
                          children [
                            Shape {{
                              appearance PBRAppearance {{ baseColor 0.7 0.5 0.3 }}
                              geometry Capsule {{ height 0.42 radius 0.06 }}
                            }}
                            # --- 右膝 ---
                            HingeJoint {{
                              jointParameters HingeJointParameters {{
                                axis 1 0 0
                                position IS knee_angle_r
                              }}
                              device [
                                RotationalMotor {{ name "knee_r_motor" maxTorque 80 }}
                              ]
                              endPoint Solid {{
                                translation 0 -0.24 0
                                children [
                                  Shape {{
                                    appearance PBRAppearance {{ baseColor 0.7 0.5 0.3 }}
                                    geometry Capsule {{ height 0.38 radius 0.05 }}
                                  }}
                                  # --- 右踝 (2-DOF) ---
                                  HingeJoint {{
                                    jointParameters HingeJointParameters {{
                                      axis 1 0 0
                                      position IS ankle_angle_r
                                    }}
                                    device [
                                      RotationalMotor {{ name "ankle_r_motor" maxTorque 30 }}
                                    ]
                                    endPoint Solid {{
                                      translation 0 -0.22 0
                                      children [
                                        Shape {{
                                          appearance PBRAppearance {{ baseColor 0.8 0.7 0.5 }}
                                          geometry Box {{ size 0.05 0.08 0.22 }}
                                        }}
                                      ]
                                    }}
                                  }}
                                ]
                              }}
                            }}
                          ]
                        }}
                      }}
                    ]
                  }}
                }}
              ]
            }}
          }}
          # --- 左髋 (对称) ---
          HingeJoint {{
            jointParameters HingeJointParameters {{
              axis 1 0 0
              position IS hip_flexion_l
            }}
            device [
              RotationalMotor {{ name "hip_flexion_l_motor" maxTorque 100 }}
            ]
            endPoint Solid {{
              translation -0.09 -0.06 0.06
              rotation 0 0 1 3.14159
              children [
                HingeJoint {{
                  jointParameters HingeJointParameters {{
                    axis 0 0 1
                    position IS hip_abduction_l
                  }}
                  device [
                    RotationalMotor {{ name "hip_abd_l_motor" maxTorque 50 }}
                  ]
                  endPoint Solid {{
                    children [
                      HingeJoint {{
                        jointParameters HingeJointParameters {{
                          axis 0 1 0
                          position IS hip_rotation_l
                        }}
                        device [
                          RotationalMotor {{ name "hip_rot_l_motor" maxTorque 30 }}
                        ]
                        endPoint Solid {{
                          children [
                            Shape {{
                              appearance PBRAppearance {{ baseColor 0.7 0.5 0.3 }}
                              geometry Capsule {{ height 0.42 radius 0.06 }}
                            }}
                            HingeJoint {{
                              jointParameters HingeJointParameters {{
                                axis 1 0 0
                                position IS knee_angle_l
                              }}
                              device [
                                RotationalMotor {{ name "knee_l_motor" maxTorque 80 }}
                              ]
                              endPoint Solid {{
                                translation 0 -0.24 0
                                children [
                                  Shape {{
                                    appearance PBRAppearance {{ baseColor 0.7 0.5 0.3 }}
                                    geometry Capsule {{ height 0.38 radius 0.05 }}
                                  }}
                                  HingeJoint {{
                                    jointParameters HingeJointParameters {{
                                      axis 1 0 0
                                      position IS ankle_angle_l
                                    }}
                                    device [
                                      RotationalMotor {{ name "ankle_l_motor" maxTorque 30 }}
                                    ]
                                    endPoint Solid {{
                                      translation 0 -0.22 0
                                      children [
                                        Shape {{
                                          appearance PBRAppearance {{ baseColor 0.8 0.7 0.5 }}
                                          geometry Box {{ size 0.05 0.08 0.22 }}
                                        }}
                                      ]
                                    }}
                                  }}
                                ]
                              }}
                            }}
                          ]
                        }}
                      }}
                    ]
                  }}
                }}
              ]
            }}
          }}
        ]
        # 骨盆几何体
        children [
          Shape {{
            appearance PBRAppearance {{ baseColor 0.8 0.8 0.8 }}
            geometry Capsule {{ height 0.18 radius 0.11 }}
          }}
        ]
      }}
    ]

    # 地面参考平面
    children [
      Solid {{
        translation 0 -0.01 0
        contactMaterial "default"
        boundingObject Transform {{
          translation 0 -0.5 0
          children [ Box {{ size 2 0.01 2 }} ]
        }}
      }}
    ]
  }}
}}
'''

    def _generate_default_proto(self, name: str) -> str:
        """生成默认PROTO文件."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{name}.proto")
        content = self._build_default_proto(name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path


def parse_osim_model(osim_path: str) -> dict:
    """解析 OpenSim .osim 文件，提取骨骼结构信息.

    Returns:
        {
            "bodies": {name: {mass, com, inertia, ...}},
            "joints": [{name, type, parent, child, ...}],
            "markers": [{name, body, location}],
        }
    """
    if not os.path.exists(osim_path):
        return {"bodies": {}, "joints": [], "markers": []}

    tree = ET.parse(osim_path)
    root = tree.getroot()

    # 解析 bodies
    bodies = {}
    body_set = root.find(".//BodySet/objects")
    if body_set is not None:
        for body in body_set:
            name = body.get("name", "")
            if not name:
                continue
            bodies[name] = {
                "mass": float(body.findtext("mass", "0")),
                "mass_center": _parse_vec3(body.find("mass_center")),
                "inertia": _parse_vec6(body.find("inertia")),
            }

    # 解析 joints
    joints = []
    joint_set = root.find(".//JointSet/objects")
    if joint_set is not None:
        for joint in joint_set:
            joint_type = joint.tag.split("}")[-1] if "}" in joint.tag else joint.tag
            joints.append({
                "name": joint.get("name", ""),
                "type": joint_type,
                "parent_body": joint.findtext("parent_body", "ground"),
                "child_body": joint.findtext("child_body", ""),
                "location_in_parent": _parse_vec3(joint.find("location_in_parent")),
                "location_in_child": _parse_vec3(joint.find("location_in_child")),
                "orientation_in_parent": _parse_vec3(joint.find("orientation_in_parent")),
            })

    return {"bodies": bodies, "joints": joints, "markers": []}


def _parse_vec3(elem) -> list[float]:
    """解析 OpenSim Vec3 元素."""
    if elem is None:
        return [0.0, 0.0, 0.0]
    text = elem.text.strip() if elem.text else ""
    parts = text.split()
    if len(parts) >= 3:
        return [float(p) for p in parts[:3]]
    return [0.0, 0.0, 0.0]


def _parse_vec6(elem) -> list[float]:
    """解析 OpenSim Vec6 元素 (惯性张量)."""
    if elem is None:
        return [0.0] * 6
    text = elem.text.strip() if elem.text else ""
    parts = text.split()
    if len(parts) >= 6:
        return [float(p) for p in parts[:6]]
    return [0.0] * 6
